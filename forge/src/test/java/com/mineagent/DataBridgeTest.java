package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class DataBridgeTest {

  private DataBridge dataBridge;
  private NetworkHandler mockNetworkHandler;

  @BeforeEach
  void setUp() {
    // Reset singleton instance by using reflection or creating fresh instance
    // Since DataBridge uses singleton pattern, we'll work with the instance
    dataBridge = DataBridge.getInstance();
    mockNetworkHandler = mock(NetworkHandler.class);
    dataBridge.takeExtrinsicReward();
  }

  @Test
  void getInstance_returnsSameInstance() {
    DataBridge instance1 = DataBridge.getInstance();
    DataBridge instance2 = DataBridge.getInstance();

    assertSame(instance1, instance2);
  }

  @Test
  void setLatestAction_andGetLatestAction_getAndSetBehavior() {
    ActionMessage input1 =
        new ActionMessage(
            ActionMessage.MSG_TYPE_ACTION, new int[] {65}, new int[0], false, 0f, 0f,
            false, 0, 0, false, 0f, "");
    ActionMessage input2 =
        new ActionMessage(
            ActionMessage.MSG_TYPE_ACTION, new int[] {66}, new int[0], false, 0f, 0f,
            false, 0, 0, false, 0f, "");

    // Set first input
    dataBridge.setLatestAction(input1);

    // Get should return and clear
    ActionMessage retrieved = dataBridge.getLatestAction();
    assertEquals(input1, retrieved);

    // Get again should return null (was cleared)
    ActionMessage retrieved2 = dataBridge.getLatestAction();
    assertNull(retrieved2);

    // Set second input
    dataBridge.setLatestAction(input2);

    // Get should return second input
    ActionMessage retrieved3 = dataBridge.getLatestAction();
    assertEquals(input2, retrieved3);
  }

  @Test
  void integration_setLatestActionThenGetAndSendObservation() {
    dataBridge.setNetworkHandler(mockNetworkHandler);

    // Set action
    ActionMessage input =
        new ActionMessage(
            ActionMessage.MSG_TYPE_ACTION, new int[] {65, 66}, new int[0], true, 10.0f,
            20.0f, true, 1, 0, true, 5.0f, "hello");
    dataBridge.setLatestAction(input);

    // Get action
    ActionMessage retrieved = dataBridge.getLatestAction();
    assertEquals(input, retrieved);

    // Send observation
    Observation obs = new Observation(100.0, new byte[] {1, 2, 3});
    dataBridge.sendObservation(obs);

    verify(mockNetworkHandler, times(1)).setLatest(obs.frame(), obs.reward());
  }

  @Test
  void extrinsicReward_accumulatesResetsAndSends() {
    dataBridge.addExtrinsicReward(-1.5);
    dataBridge.addExtrinsicReward(-2.5);
    assertTrue(dataBridge.hasPendingExtrinsicReward());
    assertEquals(-4.0, dataBridge.takeExtrinsicReward(), 1e-9);
    assertFalse(dataBridge.hasPendingExtrinsicReward());

    dataBridge.setNetworkHandler(mockNetworkHandler);
    dataBridge.addExtrinsicReward(-7.0);
    byte[] frame = new byte[] {9};
    dataBridge.sendObservation(new Observation(dataBridge.takeExtrinsicReward(), frame));
    verify(mockNetworkHandler).setLatest(frame, -7.0);
  }
}
