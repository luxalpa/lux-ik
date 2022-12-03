use bevy::prelude::*;
use bevy_egui::EguiPlugin;

pub fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(EguiPlugin)
        .run();
}
