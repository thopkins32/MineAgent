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
  }

  @Test
  void getInstance_returnsSameInstance() {
    DataBridge instance1 = DataBridge.getInstance();
    DataBridge instance2 = DataBridge.getInstance();

    assertSame(instance1, instance2);
  }

  @Test
  void setLatestRawInput_andGetLatestRawInput_getAndSetBehavior() {
    RawInput input1 = new RawInput(new int[] {65}, 1.0f, 2.0f, (byte) 1, 0.5f, "test");
    RawInput input2 = new RawInput(new int[] {66}, 3.0f, 4.0f, (byte) 2, 1.0f, "test2");

    // Set first input
    dataBridge.setLatestRawInput(input1);

    // Get should return and clear
    RawInput retrieved = dataBridge.getLatestRawInput();
    assertEquals(input1, retrieved);

    // Get again should return null (was cleared)
    RawInput retrieved2 = dataBridge.getLatestRawInput();
    assertNull(retrieved2);

    // Set second input
    dataBridge.setLatestRawInput(input2);

    // Get should return second input
    RawInput retrieved3 = dataBridge.getLatestRawInput();
    assertEquals(input2, retrieved3);
  }

  @Test
  void integration_setLatestRawInputThenGetAndSendObservation() {
    dataBridge.setNetworkHandler(mockNetworkHandler);

    // Set raw input
    RawInput input = new RawInput(new int[] {65, 66}, 10.0f, 20.0f, (byte) 1, 5.0f, "hello");
    dataBridge.setLatestRawInput(input);

    // Get raw input
    RawInput retrieved = dataBridge.getLatestRawInput();
    assertEquals(input, retrieved);

    // Send observation
    Observation obs = new Observation(100.0, new byte[] {1, 2, 3});
    dataBridge.sendObservation(obs);

    verify(mockNetworkHandler, times(1)).setLatest(obs.frame(), obs.reward());
  }
}
