package com.mineagent;

import com.mojang.logging.LogUtils;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;

/**
 * Central bridge for data exchange between network handler and game events. Manages the latest raw
 * input, observations, and connection state.
 */
public class DataBridge {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge instance;

  private NetworkHandler networkHandler;

  // Raw input handling
  private final AtomicReference<RawInput> latestRawInput = new AtomicReference<>();

  // Input injection
  private final InputInjector inputInjector = new InputInjector();

  // Connection state for input suppression
  private final AtomicBoolean clientConnected = new AtomicBoolean(false);

  private DataBridge() {}

  public static synchronized DataBridge getInstance() {
    if (instance == null) {
      instance = new DataBridge();
      LOGGER.info("DataBridge instance created");
    }
    return instance;
  }

  public void setNetworkHandler(NetworkHandler handler) {
    this.networkHandler = handler;
    LOGGER.info("NetworkHandler connected to DataBridge");
  }

  /** Sends an observation to connected clients. */
  public void sendObservation(Observation obs) {
    if (networkHandler != null) {
      networkHandler.setLatest(obs.frame(), obs.reward());
    } else {
      LOGGER.warn("Cannot send frame - NetworkHandler is null");
    }
  }

  /** Sets the latest raw input received from the Python agent. */
  public void setLatestRawInput(RawInput rawInput) {
    latestRawInput.set(rawInput);
  }

  /** Gets and clears the latest raw input. Returns null if no new input is available. */
  public RawInput getLatestRawInput() {
    RawInput rawInput = latestRawInput.getAndSet(null);
    if (rawInput != null) {
      LOGGER.debug("DataBridge getting latest raw input: {} keys", rawInput.keyCodes().length);
    }
    return rawInput;
  }

  /** Gets the input injector for injecting raw input into Minecraft. */
  public InputInjector getInputInjector() {
    return inputInjector;
  }

  /** Sets whether a Python client is connected. Used for input suppression. */
  public void setClientConnected(boolean connected) {
    boolean wasConnected = clientConnected.getAndSet(connected);
    if (wasConnected != connected) {
      LOGGER.info("Client connection state changed: {}", connected ? "CONNECTED" : "DISCONNECTED");
    }
  }

  /** Returns whether a Python client is currently connected. */
  public boolean isClientConnected() {
    return clientConnected.get();
  }
}
