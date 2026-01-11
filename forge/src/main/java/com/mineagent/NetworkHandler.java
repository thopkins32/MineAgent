package com.mineagent;

import com.mojang.logging.LogUtils;
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
 * Handles network communication between the Minecraft mod and Python agent.
 * Uses Unix domain sockets for low-latency IPC.
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
                throw new IOException("Failed to create observation socket file: " + OBSERVATION_SOCKET_PATH);
            }
            if (!Files.exists(Path.of(ACTION_SOCKET_PATH))) {
                throw new IOException("Failed to create action socket file: " + ACTION_SOCKET_PATH);
            }

            LOGGER.info("Socket files verified - Observation: {}, Action: {}",
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
     * Handles an action client connection, reading variable-size RawInput messages.
     * 
     * RawInput protocol format:
     * - 1 byte: numKeysPressed (0-255)
     * - N*2 bytes: keyCodes (shorts)
     * - 4 bytes: mouseDeltaX (float)
     * - 4 bytes: mouseDeltaY (float)
     * - 1 byte: mouseButtons
     * - 4 bytes: scrollDelta (float)
     * - 2 bytes: textLength
     * - M bytes: textBytes (UTF-8)
     * 
     * Minimum size: 16 bytes (no keys, no text)
     */
    private void handleActionClient(SocketChannel clientSocket) {
        actionExecutor.submit(() -> {
            try {
                clientSocket.configureBlocking(true);
                
                // Buffer for reading the header (1 byte for key count)
                ByteBuffer headerBuffer = ByteBuffer.allocate(1);
                
                while (this.running.get()) {
                    // Step 1: Read the key count (1 byte)
                    headerBuffer.clear();
                    if (readExact(clientSocket, headerBuffer) == -1) {
                        LOGGER.info("Action client disconnected");
                        break;
                    }
                    headerBuffer.flip();
                    int numKeys = headerBuffer.get() & 0xFF;
                    
                    // Step 2: Calculate remaining message size
                    // keyCodes(N*2) + mouseDx(4) + mouseDy(4) + mouseButtons(1) + scrollDelta(4) + textLen(2)
                    int fixedSize = (numKeys * 2) + 4 + 4 + 1 + 4 + 2;
                    ByteBuffer fixedBuffer = ByteBuffer.allocate(fixedSize);
                    
                    if (readExact(clientSocket, fixedBuffer) == -1) {
                        LOGGER.info("Action client disconnected during fixed read");
                        break;
                    }
                    fixedBuffer.flip();
                    
                    // Read key codes
                    int[] keyCodes = new int[numKeys];
                    for (int i = 0; i < numKeys; i++) {
                        keyCodes[i] = fixedBuffer.getShort();
                    }
                    
                    // Read mouse and scroll data
                    float mouseDx = fixedBuffer.getFloat();
                    float mouseDy = fixedBuffer.getFloat();
                    byte mouseButtons = fixedBuffer.get();
                    float scrollDelta = fixedBuffer.getFloat();
                    
                    // Read text length
                    int textLength = fixedBuffer.getShort() & 0xFFFF;
                    
                    // Step 3: Read text if present
                    String text = "";
                    if (textLength > 0) {
                        ByteBuffer textBuffer = ByteBuffer.allocate(textLength);
                        if (readExact(clientSocket, textBuffer) == -1) {
                            LOGGER.info("Action client disconnected during text read");
                            break;
                        }
                        textBuffer.flip();
                        byte[] textBytes = new byte[textLength];
                        textBuffer.get(textBytes);
                        text = new String(textBytes, java.nio.charset.StandardCharsets.UTF_8);
                    }
                    
                    // Create and process the RawInput
                    final RawInput rawInput = new RawInput(keyCodes, mouseDx, mouseDy, mouseButtons, scrollDelta, text);
                    processRawInput(rawInput);
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
     * Reads exactly the buffer's remaining capacity from the socket.
     * Returns -1 if the client disconnects, otherwise returns bytes read.
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
     * Processes a received RawInput by passing it to the DataBridge.
     */
    private void processRawInput(RawInput rawInput) {
        DataBridge.getInstance().setLatestRawInput(rawInput);
        LOGGER.debug("RawInput received: {} keys, mouse=({}, {}), buttons={}, scroll={}, text='{}'",
            rawInput.keyCodes().length,
            rawInput.mouseDx(), rawInput.mouseDy(),
            rawInput.mouseButtons(),
            rawInput.scrollDelta(),
            rawInput.text());
    }

    private void handleObservationClient(SocketChannel clientSocket) {
        observationExecutor.submit(() -> {
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

    /**
     * Sets the latest observation data to be sent to connected clients.
     */
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
    
    /**
     * Stops the network handler gracefully.
     */
    public void stop() {
        this.running.set(false);
    }
}
