package com.mineagent;

import com.mojang.blaze3d.platform.Window;
import com.mojang.logging.LogUtils;
import net.minecraft.client.Minecraft;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.config.ModConfig;
import net.minecraftforge.fml.event.lifecycle.FMLClientSetupEvent;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;
import net.minecraftforge.fml.loading.FMLEnvironment;
import org.slf4j.Logger;

// The value here should match an entry in the META-INF/mods.toml file
@Mod(MineAgentMod.MODID)
public class MineAgentMod {
  // Define mod id in a common place for everything to reference
  public static final String MODID = "mineagent";
  // Directly reference a slf4j logger
  private static final Logger LOGGER = LogUtils.getLogger();

  private static Thread networkThread;

  public MineAgentMod(FMLJavaModLoadingContext context) {
    // Register client event handler
    MinecraftForge.EVENT_BUS.register(ClientEventHandler.class);

    // Register our mod's ForgeConfigSpec so that Forge can create and load the config file for us
    context.registerConfig(ModConfig.Type.COMMON, Config.SPEC);

    // Register for MOD bus events (lifecycle events)
    context.getModEventBus().register(this);
  }

  @SubscribeEvent
  public void onClientSetup(FMLClientSetupEvent event) {
    // Only run on client side
    if (FMLEnvironment.dist == Dist.CLIENT) {
      event.enqueueWork(
          () -> {
            setupWindow();
            setupClientNetworking();
          });
    }
  }

  private void setupWindow() {
    Minecraft mc = Minecraft.getInstance();
    if (mc.getWindow() != null) {
      Window window = mc.getWindow();
      int configWidth = Config.WINDOW_WIDTH.get();
      int configHeight = Config.WINDOW_HEIGHT.get();

      // Only resize if different from current size
      if (window.getWidth() != configWidth || window.getHeight() != configHeight) {
        window.setWindowed(configWidth, configHeight);
        LOGGER.info("Window resized to {}x{}", configWidth, configHeight);
      } else {
        LOGGER.info("Window already set to {}x{}", configWidth, configHeight);
      }
    }
  }

  private void setupClientNetworking() {
    if (networkThread == null || !networkThread.isAlive()) {
      LOGGER.info("Starting client-side network handler");
      NetworkHandler networkHandler = new NetworkHandler();
      DataBridge.getInstance().setNetworkHandler(networkHandler);
      networkThread = new Thread(networkHandler);
      networkThread.start();
      LOGGER.info(
          "Client network handler started on TCP port "
              + Config.READ_PORT.get()
              + " and UDP port "
              + Config.WRITE_PORT.get());

      // Add a shutdown hook to properly cleanup socket files
      Runtime.getRuntime()
          .addShutdownHook(
              new Thread(
                  () -> {
                    LOGGER.info("Shutting down client-side network handler");
                    if (networkThread != null) {
                      networkThread.interrupt();
                      try {
                        networkThread.join(5000);
                      } catch (InterruptedException e) {
                        LOGGER.error("Interrupted while waiting for network thread to finish", e);
                      }
                    }
                  }));
    }
  }
}
