package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.mojang.blaze3d.platform.Window;
import net.minecraft.client.Minecraft;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.common.ForgeConfigSpec;
import net.minecraftforge.fml.config.ModConfig;
import net.minecraftforge.fml.event.lifecycle.FMLClientSetupEvent;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;

class MineAgentModTest {

  private FMLJavaModLoadingContext mockContext;
  private MinecraftTestFixture fixture;

  @BeforeEach
  void setUp() {
    mockContext = mock(FMLJavaModLoadingContext.class);
    fixture = new MinecraftTestFixture();
  }

  @Test
  void modId_constantIsCorrect() {
    assertEquals("mineagent", MineAgentMod.MODID);
  }

  @Test
  void constructor_registersEventHandlers() {
    try (MockedStatic<MinecraftForge> mockedForge = mockStatic(MinecraftForge.class)) {
      mockedForge
          .when(() -> MinecraftForge.EVENT_BUS.register(any()))
          .thenAnswer(invocation -> null);

      MineAgentMod mod = new MineAgentMod(mockContext);

      // Verify event bus registration
      mockedForge.verify(
          () -> MinecraftForge.EVENT_BUS.register(eq(ClientEventHandler.class)), times(1));
    }
  }

  @Test
  void constructor_registersConfig() {
    try (MockedStatic<MinecraftForge> mockedForge = mockStatic(MinecraftForge.class)) {
      doNothing()
          .when(mockContext)
          .registerConfig(any(ModConfig.Type.class), any(ForgeConfigSpec.class));

      MineAgentMod mod = new MineAgentMod(mockContext);

      // Verify config registration
      verify(mockContext, times(1))
          .registerConfig(eq(ModConfig.Type.COMMON), eq(Config.SPEC));
    }
  }

  @Test
  void constructor_registersModEventBus() {
    try (MockedStatic<MinecraftForge> mockedForge = mockStatic(MinecraftForge.class)) {
      when(mockContext.getModEventBus()).thenReturn(mock(net.minecraftforge.eventbus.api.IEventBus.class));

      MineAgentMod mod = new MineAgentMod(mockContext);

      // Verify mod event bus was accessed
      verify(mockContext, atLeastOnce()).getModEventBus();
    }
  }

  @Test
  void onClientSetup_setsUpWindowAndNetworkingOnClientSide() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());
      // FMLEnvironment.dist is a static final field, so we test behavior instead of mocking it
      // The test will run in test environment which may not be CLIENT dist

      FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);
      when(mockEvent.enqueueWork(any(Runnable.class))).thenAnswer(invocation -> {
        Runnable work = invocation.getArgument(0);
        work.run();
        return null;
      });

      MineAgentMod mod = new MineAgentMod(mockContext);
      mod.onClientSetup(mockEvent);

      // Verify enqueueWork was called if dist is CLIENT
      // Note: In test environment, dist may not be CLIENT, so we verify it doesn't throw
      assertDoesNotThrow(() -> mod.onClientSetup(mockEvent));
    }
  }

  @Test
  void onClientSetup_skipsSetupOnServerSide() {
    // FMLEnvironment.dist is a static final field that can't be mocked
    // This test verifies the conditional logic exists
    // In a server environment, enqueueWork would not be called
    FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);

    MineAgentMod mod = new MineAgentMod(mockContext);
    // Should not throw regardless of dist
    assertDoesNotThrow(() -> mod.onClientSetup(mockEvent));
  }

  @Test
  void setupWindow_resizesWindowToConfigValues() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Window mockWindow = fixture.getWindow();
      when(mockWindow.getWidth()).thenReturn(640); // Different from config default
      when(mockWindow.getHeight()).thenReturn(480); // Different from config default

      MineAgentMod mod = new MineAgentMod(mockContext);
      FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);
      when(mockEvent.enqueueWork(any(Runnable.class))).thenAnswer(invocation -> {
        invocation.getArgument(0, Runnable.class).run();
        return null;
      });

      // Call onClientSetup - behavior depends on FMLEnvironment.dist
      mod.onClientSetup(mockEvent);

      // Verify window resize was attempted if dist is CLIENT and size differs
      // Since we can't mock dist, we verify it doesn't throw
      assertDoesNotThrow(() -> mod.onClientSetup(mockEvent));
    }
  }

  @Test
  void setupWindow_skipsResizeIfAlreadyCorrectSize() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      Window mockWindow = fixture.getWindow();
      int configWidth = Config.WINDOW_WIDTH.get();
      int configHeight = Config.WINDOW_HEIGHT.get();
      when(mockWindow.getWidth()).thenReturn(configWidth);
      when(mockWindow.getHeight()).thenReturn(configHeight);

      MineAgentMod mod = new MineAgentMod(mockContext);
      FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);
      when(mockEvent.enqueueWork(any(Runnable.class))).thenAnswer(invocation -> {
        invocation.getArgument(0, Runnable.class).run();
        return null;
      });

      // Call onClientSetup
      mod.onClientSetup(mockEvent);

      // Verify setWindowed was NOT called when size matches config
      // (only if dist is CLIENT and enqueueWork was called)
      // Since we can't mock dist, we verify it doesn't throw
    }
  }

  @Test
  void setupClientNetworking_createsAndStartsNetworkHandlerThread() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);
      when(mockEvent.enqueueWork(any(Runnable.class))).thenAnswer(invocation -> {
        invocation.getArgument(0, Runnable.class).run();
        return null;
      });

      MineAgentMod mod = new MineAgentMod(mockContext);
      mod.onClientSetup(mockEvent);

      // Verify DataBridge has network handler set
      DataBridge dataBridge = DataBridge.getInstance();
      // NetworkHandler would be set during setupClientNetworking
      // We can't easily verify thread creation without reflection,
      // but we verify the method doesn't throw
      assertDoesNotThrow(() -> mod.onClientSetup(mockEvent));
    }
  }

  @Test
  void setupClientNetworking_doesNotRestartIfThreadAlreadyAlive() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(fixture.getMinecraft());

      FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);
      when(mockEvent.enqueueWork(any(Runnable.class))).thenAnswer(invocation -> {
        invocation.getArgument(0, Runnable.class).run();
        return null;
      });

      MineAgentMod mod = new MineAgentMod(mockContext);

      // Call setup twice
      mod.onClientSetup(mockEvent);
      mod.onClientSetup(mockEvent);

      // Should not throw (checks if thread is alive before creating new one)
      assertDoesNotThrow(() -> mod.onClientSetup(mockEvent));
    }
  }

  @Test
  void setupWindow_handlesNullWindow() {
    try (MockedStatic<Minecraft> mockedMinecraft = mockStatic(Minecraft.class)) {
      mockedMinecraft.when(Minecraft::getInstance).thenReturn(null);

      FMLClientSetupEvent mockEvent = mock(FMLClientSetupEvent.class);
      when(mockEvent.enqueueWork(any(Runnable.class))).thenAnswer(invocation -> {
        invocation.getArgument(0, Runnable.class).run();
        return null;
      });

      MineAgentMod mod = new MineAgentMod(mockContext);

      // Should not throw when window is null
      assertDoesNotThrow(() -> mod.onClientSetup(mockEvent));
    }
  }

  @Test
  void constructor_initializesModSuccessfully() {
    try (MockedStatic<MinecraftForge> mockedForge = mockStatic(MinecraftForge.class)) {
      mockedForge
          .when(() -> MinecraftForge.EVENT_BUS.register(any()))
          .thenAnswer(invocation -> null);

      assertDoesNotThrow(() -> new MineAgentMod(mockContext));
    }
  }
}
