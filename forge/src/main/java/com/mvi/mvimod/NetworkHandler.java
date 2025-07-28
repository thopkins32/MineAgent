package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.StandardProtocolFamily;
import java.net.StandardSocketOptions;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;

public class NetworkHandler implements Runnable {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static final String SEND_SOCKET_PATH = "/tmp/mvi_send.sock";
  private static final String RECEIVE_SOCKET_PATH = "/tmp/mvi_receive.sock";
  private static final ExecutorService senderExecutor = Executors.newCachedThreadPool();
  private static final ExecutorService receiverExecutor = Executors.newCachedThreadPool();
  private Thread sendThread;
  private Thread receiveThread;
  private ServerSocketChannel sendSocketChannel;
  private ServerSocketChannel receiveSocketChannel;
  private final AtomicBoolean running = new AtomicBoolean(true);

  // Async observation sending
  private final AtomicReference<Observation> latestObservation = new AtomicReference<Observation>();
  private final Semaphore frameAvailable = new Semaphore(0);

  // Observation data container
  private static class Observation {
    final byte[] frameBuffer;
    final int reward;
    final long timestamp;

    Observation(byte[] frameBuffer, int reward) {
      this.frameBuffer = frameBuffer;
      this.reward = reward;
      this.timestamp = System.currentTimeMillis();
    }
  }

  @Override
  public void run() {
    try {
      Files.deleteIfExists(Path.of(SEND_SOCKET_PATH));
      Files.deleteIfExists(Path.of(RECEIVE_SOCKET_PATH));

      sendSocketChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
      sendSocketChannel.bind(UnixDomainSocketAddress.of(SEND_SOCKET_PATH));
      sendSocketChannel.configureBlocking(true);

      receiveSocketChannel = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
      receiveSocketChannel.bind(UnixDomainSocketAddress.of(RECEIVE_SOCKET_PATH));
      receiveSocketChannel.configureBlocking(true);

      LOGGER.info("Socket files created: {} and {}", SEND_SOCKET_PATH, RECEIVE_SOCKET_PATH);

      // Verify socket files were actually created
      if (!Files.exists(Path.of(SEND_SOCKET_PATH))) {
        throw new IOException("Failed to create send socket file: " + SEND_SOCKET_PATH);
      }
      if (!Files.exists(Path.of(RECEIVE_SOCKET_PATH))) {
        throw new IOException("Failed to create receive socket file: " + RECEIVE_SOCKET_PATH);
      }

      LOGGER.info(
          "Socket files verified - Send: {}, Receive: {}",
          Files.exists(Path.of(SEND_SOCKET_PATH)),
          Files.exists(Path.of(RECEIVE_SOCKET_PATH)));

      sendThread = new Thread(this::acceptSendClients, "SendClients");
      receiveThread = new Thread(this::acceptReceiveClients, "ReceiveClients");

      sendThread.start();
      receiveThread.start();

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

  private void acceptSendClients() {
    LOGGER.info("Send clients acceptor thread started");
    while (this.running.get() && !Thread.currentThread().isInterrupted()) {
      try {
        SocketChannel clientSocket = sendSocketChannel.accept();
        LOGGER.info("Send client connected: " + clientSocket.getRemoteAddress());
        // 1MB send buffer which can fit small frames
        clientSocket.setOption(StandardSocketOptions.SO_SNDBUF, 1024 * 1024);
        handleSendClient(clientSocket);
      } catch (IOException e) {
        if (this.running.get()) {
          LOGGER.error("Error accepting send client", e);
        } else {
          LOGGER.info("Send socket channel closed, stopping accept loop");
          break;
        }
      }
    }
    LOGGER.info("Send clients acceptor thread stopped");
  }

  private void acceptReceiveClients() {
    LOGGER.info("Receive clients acceptor thread started");
    while (this.running.get() && !Thread.currentThread().isInterrupted()) {
      try {
        SocketChannel clientSocket = receiveSocketChannel.accept();
        LOGGER.info("Receive client connected: " + clientSocket.getRemoteAddress());
        handleReceiveClient(clientSocket);
      } catch (IOException e) {
        if (this.running.get()) {
          LOGGER.error("Error accepting receive client", e);
        } else {
          LOGGER.info("Receive socket channel closed, stopping accept loop");
          break;
        }
      }
    }
    LOGGER.info("Receive clients acceptor thread stopped");
  }

  private void handleReceiveClient(SocketChannel clientSocket) {
    receiverExecutor.submit(
        () -> {
          try {
            clientSocket.configureBlocking(true);
            ByteBuffer actionBuffer = ByteBuffer.allocate(11); // Action is exactly 11 bytes
            
            while (this.running.get()) {
              actionBuffer.clear();
              
              // Read exactly 11 bytes for one Action
              int totalBytesRead = 0;
              while (totalBytesRead < 11) {
                int bytesRead = clientSocket.read(actionBuffer);
                if (bytesRead == -1) {
                  // Client disconnected
                  LOGGER.info("Client disconnected");
                  return;
                }
                totalBytesRead += bytesRead;
              }

              // Process each action asynchronously to avoid blocking
              CompletableFuture.runAsync(() -> processCommand(actionBuffer), receiverExecutor)
                  .exceptionally(
                      throwable -> {
                        LOGGER.error("Error processing action: " + actionBuffer, throwable);
                        return null;
                      });
            }
          } catch (IOException e) {
            LOGGER.error("Error handling client", e);
          } finally {
            try {
              clientSocket.close();
            } catch (IOException e) {
              LOGGER.error("Error closing client socket", e);
            }
          }
        });
  }

  private void handleSendClient(SocketChannel clientSocket) {
    senderExecutor.submit(
        () -> {
          try {
            while (this.running.get()) {
              try {
                frameAvailable.acquire();
                Observation observation = latestObservation.get();
                if (observation != null) {
                  sendObservationImmediate(observation, clientSocket);
                }
              } catch (InterruptedException e) {
                LOGGER.error("Interrupted while waiting for frame", e);
                this.running.set(false);
              }
            }
          } finally {
            try {
              clientSocket.close();
            } catch (IOException e) {
              LOGGER.error("Error closing client socket", e);
            }
          }
        });
  }

  public void setLatest(byte[] frameBuffer, int reward) {
    Observation observation = new Observation(frameBuffer, reward);
    latestObservation.set(observation);
    frameAvailable.drainPermits();
    frameAvailable.release();
  }

  private void sendObservationImmediate(Observation observation, SocketChannel clientSocket) {
    try {
      int totalSize = 8 + observation.frameBuffer.length;
      ByteBuffer buffer = ByteBuffer.allocate(totalSize);
      buffer.putInt(observation.reward);
      buffer.putInt(observation.frameBuffer.length);
      buffer.put(observation.frameBuffer);
      buffer.flip();
      while (buffer.hasRemaining()) {
        int bytesWritten = clientSocket.write(buffer);
        LOGGER.info("Observation sent: {} bytes", bytesWritten);
      }
    } catch (IOException e) {
      LOGGER.error("Error sending observation", e);
    }
  }

  private void processCommand(ByteBuffer actionBuffer) {
    final Action action = Action.fromBytes(actionBuffer.array());
    DataBridge.getInstance().setLatestAction(action);
  }

  private void cleanup() {
    LOGGER.info("Shutting down NetworkHandler...");
    this.running.set(false);
    
    // Release all pressed keys on cleanup
    ClientEventHandler.releaseAllKeys();
    // Wait for threads to finish
    if (this.sendThread != null) {
      try {
        this.sendThread.interrupt();
        this.sendThread.join(5000); // Wait up to 5 seconds
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }

    if (this.receiveThread != null) {
      try {
        this.receiveThread.interrupt();
        this.receiveThread.join(5000); // Wait up to 5 seconds
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }

    // Shutdown executors
    senderExecutor.shutdown();
    receiverExecutor.shutdown();

    try {
      if (this.sendSocketChannel != null) {
        this.sendSocketChannel.close();
      }
      if (this.receiveSocketChannel != null) {
        this.receiveSocketChannel.close();
      }
    } catch (IOException e) {
      LOGGER.error("Cleanup error: " + e.getMessage());
    }

    try {
      Files.deleteIfExists(Path.of(SEND_SOCKET_PATH));
      Files.deleteIfExists(Path.of(RECEIVE_SOCKET_PATH));
    } catch (IOException e) {
      LOGGER.error("Cleanup error: " + e.getMessage());
    }

    LOGGER.info("NetworkHandler shutdown complete");
  }
}
