package com.mvi.mvimod;

import com.mojang.logging.LogUtils;
import org.slf4j.Logger;
import java.util.concurrent.atomic.AtomicReference;

public class DataBridge {
  private static final Logger LOGGER = LogUtils.getLogger();
  private static DataBridge instance;
  private NetworkHandler networkHandler;
  
  private final AtomicReference<Action> latestAction = new AtomicReference<Action>();
  
  private DataBridge() {}

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

  public void sendObservation(Observation obs) {
    if (networkHandler != null) {
      // LOGGER.info("DataBridge sending frame data (size: {} bytes)", obs.frame().length);
      networkHandler.setLatest(obs.frame(), obs.reward());
    } else {
      LOGGER.warn("Cannot send frame - NetworkHandler is null");
    }
  }
  
  public void setLatestAction(Action action) {
    latestAction.set(action);
  }

  public Action getLatestAction() {
    Action action = latestAction.getAndSet(null);
    if (action != null) {
      LOGGER.info("DataBridge getting latest action: {}", action);
    }
    return action;
  }
}
