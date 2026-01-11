package com.mineagent;

import com.mojang.logging.LogUtils;
import org.slf4j.Logger;

public class DataBridge {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge instance;
  private NetworkHandler networkHandler;

  public static DataBridge getInstance() {
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

  public void sendEvent(String eventType, String data) {
    if (networkHandler != null) {
      LOGGER.info("Sending event: {} with data: {}", eventType, data);
      // TODO: Implement this
    } else {
      LOGGER.warn("Cannot send event - NetworkHandler is null");
    }
  }

  public void sendFrame(byte[] frameData) {
    if (networkHandler != null) {
      LOGGER.info("DataBridge sending frame data (size: {} bytes)", frameData.length);
      networkHandler.setLatest(frameData, 0);
    } else {
      LOGGER.warn("Cannot send frame - NetworkHandler is null");
    }
  }
}
