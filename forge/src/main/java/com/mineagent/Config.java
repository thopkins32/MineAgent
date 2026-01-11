package com.mineagent;

import net.minecraftforge.common.ForgeConfigSpec;
import net.minecraftforge.fml.common.Mod;

@Mod.EventBusSubscriber(modid = MineAgentMod.MODID, bus = Mod.EventBusSubscriber.Bus.MOD)
public class Config {

  // Configuration Builder
  private static final ForgeConfigSpec.Builder BUILDER = new ForgeConfigSpec.Builder();

  // Configuration Values
  public static final ForgeConfigSpec.ConfigValue<Integer> READ_PORT;
  public static final ForgeConfigSpec.ConfigValue<Integer> WRITE_PORT;
  public static final ForgeConfigSpec.ConfigValue<Integer> WINDOW_WIDTH;
  public static final ForgeConfigSpec.ConfigValue<Integer> WINDOW_HEIGHT;
  public static final ForgeConfigSpec.ConfigValue<Double> JPEG_QUALITY;
  public static final ForgeConfigSpec.ConfigValue<Integer> MAX_FRAME_SIZE;
  public static final ForgeConfigSpec.ConfigValue<Integer> CHANGE_THRESHOLD;
  public static final ForgeConfigSpec.ConfigValue<Boolean> SUPPRESS_SYSTEM_MOUSE_INPUT;

  // Built Configuration Specification
  public static final ForgeConfigSpec SPEC;

  // Static initialization block ensures proper ordering
  static {
    // Network Configuration
    BUILDER.comment("Network Configuration");
    BUILDER.push("network");

    READ_PORT =
        BUILDER
            .comment("Port for reading data from MineAgent client")
            .defineInRange("read_port", 12345, 1024, 65535);
    WRITE_PORT =
        BUILDER
            .comment("Port for sending data to MineAgent client")
            .defineInRange("write_port", 12346, 1024, 65535);

    BUILDER.pop();

    // Window Configuration
    BUILDER.comment("Window Configuration");
    BUILDER.push("window");

    WINDOW_WIDTH =
        BUILDER
            .comment("Default window width for Minecraft client")
            .defineInRange("width", 320, 320, 1920);
    WINDOW_HEIGHT =
        BUILDER
            .comment("Default window height for Minecraft client")
            .defineInRange("height", 240, 240, 1080);

    BUILDER.pop();

    // Image Compression Configuration
    BUILDER.comment("Image Compression Configuration");
    BUILDER.push("compression");

    JPEG_QUALITY =
        BUILDER
            .comment(
                "JPEG compression quality (0.1 = high compression/low quality, 1.0 = low compression/high quality)")
            .defineInRange("jpeg_quality", 0.7, 0.1, 1.0);
    MAX_FRAME_SIZE =
        BUILDER
            .comment("Maximum frame size in bytes for UDP transmission")
            .defineInRange("max_frame_size", 60000, 10000, 65000);
    CHANGE_THRESHOLD =
        BUILDER
            .comment("Pixel change threshold for delta encoding (sum of RGB differences)")
            .defineInRange("change_threshold", 10, 1, 255);

    BUILDER.pop();

    // Input Configuration
    BUILDER.comment("Input Configuration");
    BUILDER.push("input");

    SUPPRESS_SYSTEM_MOUSE_INPUT =
        BUILDER
            .comment("If true, disables OS cursor while a GUI is open and uses a virtual mouse for agent control")
            .define("suppress_system_mouse_input", true);

    BUILDER.pop();

    // Build the specification after all values are defined
    SPEC = BUILDER.build();
  }
}
