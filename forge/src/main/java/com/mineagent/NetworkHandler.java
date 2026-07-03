package com.mineagent;

import com.mojang.logging.LogUtils;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.StandardProtocolFamily;
import java.net.StandardSocketOptions;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;

/**
 * Handles network communication between the Minecraft mod and Python agent. Uses Unix domain
 * sockets for low-latency IPC.
 */
public class NetworkHandler implements Runnable {
  private static final Logger LOGGER = LogUtils.getLogger();

  // Socket paths for Unix domain sockets
  private static final String OBSERVATION_SOCKET_PATH = "/tmp/mineagent_observation.sock";
  private static final String ACTION_SOCKET_PATH = "/tmp/mineagent_action.sock";

  // Thread pool for handling clients
  private static final ExecutorService observationExecutor = Executors.newCachedThreadPool();
  private static final ExecutorService actionExecutor = Executors.newCachedThreadPool();

  // Server socket channels
  private ServerSocketChannel observationSocketChannel;
  private ServerSocketChannel actionSocketChannel;

  // Client handling threads
  private Thread observationThread;
  private Thread actionThread;

  // Running state
  private final AtomicBoolean running = new AtomicBoolean(true);

  // Async observation sending
  private final AtomicReference<ObservationData> latestObservation = new AtomicReference<>();
  private final Semaphore frameAvailable = new Semaphore(0);

  // Internal observation data container
  private static class ObservationData {
    final byte[] frameBuffer;
    final double reward;
    final long timestamp;

    ObservationData(byte[] frameBuffer, double reward) {
      this.frameBuffer = frameBuffer;
      this.reward = reward;
      this.timestamp = System.currentTimeMillis();
    }
  }

  @Override
  public void run() {
    try {
      // Clean up any existing socket files
      Files.deleteIfExists(Path.of(OBSERVATION_SOCKET_PATH));
      Files.deleteIfExists(Path.of(ACTION_SOCKET_PATH));

      // Create observation socket
      observationSocketChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
      observationSocketChannel.bind(UnixDomainSocketAddress.of(OBSERVATION_SOCKET_PATH));
      observationSocketChannel.configureBlocking(true);

      // Create action socket
      actionSocketChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
      actionSocketChannel.bind(UnixDomainSocketAddress.of(ACTION_SOCKET_PATH));
      actionSocketChannel.configureBlocking(true);

      LOGGER.info("Socket files created: {} and {}", OBSERVATION_SOCKET_PATH, ACTION_SOCKET_PATH);

      // Verify socket files were created
      if (!Files.exists(Path.of(OBSERVATION_SOCKET_PATH))) {
        throw new IOException(
            "Failed to create observation socket file: " + OBSERVATION_SOCKET_PATH);
      }
      if (!Files.exists(Path.of(ACTION_SOCKET_PATH))) {
        throw new IOException("Failed to create action socket file: " + ACTION_SOCKET_PATH);
      }

      LOGGER.info(
          "Socket files verified - Observation: {}, Action: {}",
          Files.exists(Path.of(OBSERVATION_SOCKET_PATH)),
          Files.exists(Path.of(ACTION_SOCKET_PATH)));

      // Start client acceptor threads
      observationThread = new Thread(this::acceptObservationClients, "ObservationClients");
      actionThread = new Thread(this::acceptActionClients, "ActionClients");

      observationThread.start();
      actionThread.start();

      // Keep main thread alive while server is running
      while (this.running.get() && !Thread.currentThread().isInterrupted()) {
        try {
          Thread.sleep(1000);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          break;
        }
      }

    } catch (IOException e) {
      LOGGER.error("Error starting network server", e);
    } finally {
      this.cleanup();
    }
  }

  private void acceptObservationClients() {
    LOGGER.info("Observation clients acceptor thread started");
    while (this.running.get() && !Thread.currentThread().isInterrupted()) {
      try {
        SocketChannel clientSocket = observationSocketChannel.accept();
        LOGGER.info("Observation client connected: {}", clientSocket.getRemoteAddress());
        // 1MB send buffer for frames
        clientSocket.setOption(StandardSocketOptions.SO_SNDBUF, 1024 * 1024);
        handleObservationClient(clientSocket);
      } catch (IOException e) {
        if (this.running.get()) {
          LOGGER.error("Error accepting observation client", e);
        } else {
          LOGGER.info("Observation socket channel closed, stopping accept loop");
          break;
        }
      }
    }
    LOGGER.info("Observation clients acceptor thread stopped");
  }

  private void acceptActionClients() {
    LOGGER.info("Action clients acceptor thread started");
    while (this.running.get() && !Thread.currentThread().isInterrupted()) {
      try {
        SocketChannel clientSocket = actionSocketChannel.accept();
        LOGGER.info("Action client connected: {}", clientSocket.getRemoteAddress());

        // Mark client as connected for input suppression
        DataBridge.getInstance().setClientConnected(true);

        handleActionClient(clientSocket);
      } catch (IOException e) {
        if (this.running.get()) {
          LOGGER.error("Error accepting action client", e);
        } else {
          LOGGER.info("Action socket channel closed, stopping accept loop");
          break;
        }
      }
    }
    LOGGER.info("Action clients acceptor thread stopped");
  }

  /**
   * Handles an action client connection, reading variable-size v2 ActionMessage frames.
   *
   * <p>Each message starts with a 1-byte flags header (message type + presence bits for keys /
   * mouse / buttons / scroll). The body that follows depends on which flags are set; sections are
   * read in the fixed order keys -> mouse -> buttons -> scroll, then TEXT body for TEXT messages.
   * RESET and PING carry no body.
   */
  private void handleActionClient(SocketChannel clientSocket) {
    actionExecutor.submit(
        () -> {
          try {
            clientSocket.configureBlocking(true);

            while (this.running.get()) {
              // Step 1: read the 1-byte flags header.
              ByteBuffer flagBuffer = ByteBuffer.allocate(1);
              if (readExact(clientSocket, flagBuffer) == -1) {
                LOGGER.info("Action client disconnected");
                break;
              }
              flagBuffer.flip();
              int flags = flagBuffer.get() & 0xFF;

              if ((flags & ActionMessage.FLAG_MASK_RESERVED) != 0) {
                LOGGER.warn(
                    "Invalid action message: reserved flag bits set ({}); closing stream",
                    Integer.toHexString(flags));
                break;
              }
              int msgType = flags & 0x3;

              // Step 2: read the variable body, accumulating the full message so
              // ActionMessage.fromBytes can parse it in one place.
              ByteArrayOutputStream acc = new ByteArrayOutputStream();
              acc.write(flags);

              if (msgType == ActionMessage.MSG_TYPE_ACTION) {
                if ((flags & ActionMessage.FLAG_HAS_KEYS) != 0
                    && !readKeysSection(clientSocket, acc)) {
                  break;
                }
                if ((flags & ActionMessage.FLAG_HAS_MOUSE) != 0
                    && !readFixed(clientSocket, acc, 8)) {
                  break;
                }
                if ((flags & ActionMessage.FLAG_HAS_BUTTONS) != 0
                    && !readFixed(clientSocket, acc, 1)) {
                  break;
                }
                if ((flags & ActionMessage.FLAG_HAS_SCROLL) != 0
                    && !readFixed(clientSocket, acc, 4)) {
                  break;
                }
              } else if (msgType == ActionMessage.MSG_TYPE_TEXT) {
                if (!readTextSection(clientSocket, acc)) {
                  break;
                }
              }
              // RESET / PING: no body.

              ActionMessage message;
              try {
                message = ActionMessage.fromBytes(acc.toByteArray());
              } catch (IllegalArgumentException e) {
                LOGGER.warn("Malformed action message: {}", e.getMessage());
                continue;
              }
              processAction(message);
            }
          } catch (IOException e) {
            if (this.running.get()) {
              LOGGER.error("Error handling action client", e);
            }
          } finally {
            // Mark client as disconnected
            DataBridge.getInstance().setClientConnected(false);

            // Reset input state on disconnect
            DataBridge.getInstance().getInputInjector().reset();

            try {
              clientSocket.close();
            } catch (IOException e) {
              LOGGER.error("Error closing action client socket", e);
            }
          }
        });
  }

  /**
   * Reads the keys section (u8 numPress, u8 numRelease, then the press and release key codes) and
   * appends it to {@code acc}. Returns false on disconnect.
   */
  private boolean readKeysSection(SocketChannel channel, ByteArrayOutputStream acc)
      throws IOException {
    ByteBuffer counts = ByteBuffer.allocate(2);
    if (readExact(channel, counts) == -1) {
      return false;
    }
    counts.flip();
    int numPress = counts.get() & 0xFF;
    int numRelease = counts.get() & 0xFF;
    acc.write(counts.array(), 0, 2);

    int keysBytes = (numPress + numRelease) * 2;
    ByteBuffer keys = ByteBuffer.allocate(keysBytes);
    if (readExact(channel, keys) == -1) {
      return false;
    }
    keys.flip();
    acc.write(keys.array(), 0, keysBytes);
    return true;
  }

  /** Reads the TEXT section (u16 length + UTF-8 bytes) and appends it to {@code acc}. */
  private boolean readTextSection(SocketChannel channel, ByteArrayOutputStream acc)
      throws IOException {
    ByteBuffer lenBuf = ByteBuffer.allocate(2);
    if (readExact(channel, lenBuf) == -1) {
      return false;
    }
    lenBuf.flip();
    int textLen = lenBuf.getShort() & 0xFFFF;
    acc.write(lenBuf.array(), 0, 2);

    ByteBuffer textBuf = ByteBuffer.allocate(textLen);
    if (readExact(channel, textBuf) == -1) {
      return false;
    }
    textBuf.flip();
    acc.write(textBuf.array(), 0, textLen);
    return true;
  }

  /** Reads {@code size} bytes and appends them to {@code acc}. Returns false on disconnect. */
  private boolean readFixed(SocketChannel channel, ByteArrayOutputStream acc, int size)
      throws IOException {
    ByteBuffer buf = ByteBuffer.allocate(size);
    if (readExact(channel, buf) == -1) {
      return false;
    }
    buf.flip();
    acc.write(buf.array(), 0, size);
    return true;
  }

  /**
   * Reads exactly the buffer's remaining capacity from the socket. Returns -1 if the client
   * disconnects, otherwise returns bytes read.
   */
  private int readExact(SocketChannel channel, ByteBuffer buffer) throws IOException {
    int totalRead = 0;
    while (buffer.hasRemaining()) {
      int bytesRead = channel.read(buffer);
      if (bytesRead == -1) {
        return -1;
      }
      totalRead += bytesRead;
    }
    return totalRead;
  }

  /**
   * Routes a parsed ActionMessage. ACTION / TEXT / RESET are queued through DataBridge so they are
   * applied on the client tick thread; PING is a liveness heartbeat and is dropped here.
   */
  private void processAction(ActionMessage message) {
    if (message.msgType() == ActionMessage.MSG_TYPE_PING) {
      LOGGER.debug("PING received");
      return;
    }
    DataBridge.getInstance().setLatestAction(message);
    LOGGER.debug(
        "ActionMessage received: type={}, press={}, release={}, mouse=({},{}), "
            + "buttons p={}/r={}, scroll={}, text='{}'",
        message.msgType(),
        message.keyPress().length,
        message.keyRelease().length,
        message.mouseDx(),
        message.mouseDy(),
        message.buttonPress(),
        message.buttonRelease(),
        message.scroll(),
        message.text());
  }

  private void handleObservationClient(SocketChannel clientSocket) {
    observationExecutor.submit(
        () -> {
          try {
            while (this.running.get()) {
              try {
                frameAvailable.acquire();
                ObservationData observation = latestObservation.get();
                if (observation != null) {
                  sendObservationImmediate(observation, clientSocket);
                }
              } catch (InterruptedException e) {
                LOGGER.info("Observation thread interrupted");
                Thread.currentThread().interrupt();
                break;
              }
            }
          } finally {
            try {
              clientSocket.close();
            } catch (IOException e) {
              LOGGER.error("Error closing observation client socket", e);
            }
          }
        });
  }

  /** Sets the latest observation data to be sent to connected clients. */
  public void setLatest(byte[] frameBuffer, double reward) {
    ObservationData observation = new ObservationData(frameBuffer, reward);
    latestObservation.set(observation);
    frameAvailable.drainPermits();
    frameAvailable.release();
  }

  private void sendObservationImmediate(ObservationData observation, SocketChannel clientSocket) {
    try {
      // Format: reward(8) + frameLength(4) + frame(N)
      int totalSize = 8 + 4 + observation.frameBuffer.length;
      ByteBuffer buffer = ByteBuffer.allocate(totalSize);
      buffer.putDouble(observation.reward);
      buffer.putInt(observation.frameBuffer.length);
      buffer.put(observation.frameBuffer);
      buffer.flip();

      while (buffer.hasRemaining()) {
        clientSocket.write(buffer);
      }
      LOGGER.debug("Observation sent: {} bytes", totalSize);
    } catch (IOException e) {
      LOGGER.error("Error sending observation", e);
    }
  }

  private void cleanup() {
    LOGGER.info("Shutting down NetworkHandler...");
    this.running.set(false);

    // Reset input state on cleanup
    DataBridge.getInstance().getInputInjector().reset();
    DataBridge.getInstance().setClientConnected(false);

    // Wait for threads to finish
    if (this.observationThread != null) {
      try {
        this.observationThread.interrupt();
        this.observationThread.join(5000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }

    if (this.actionThread != null) {
      try {
        this.actionThread.interrupt();
        this.actionThread.join(5000);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }

    // Shutdown executors
    observationExecutor.shutdown();
    actionExecutor.shutdown();

    // Close socket channels
    try {
      if (this.observationSocketChannel != null) {
        this.observationSocketChannel.close();
      }
      if (this.actionSocketChannel != null) {
        this.actionSocketChannel.close();
      }
    } catch (IOException e) {
      LOGGER.error("Error closing socket channels: {}", e.getMessage());
    }

    // Delete socket files
    try {
      Files.deleteIfExists(Path.of(OBSERVATION_SOCKET_PATH));
      Files.deleteIfExists(Path.of(ACTION_SOCKET_PATH));
    } catch (IOException e) {
      LOGGER.error("Error deleting socket files: {}", e.getMessage());
    }

    LOGGER.info("NetworkHandler shutdown complete");
  }

  /** Stops the network handler gracefully. */
  public void stop() {
    this.running.set(false);
  }
}
