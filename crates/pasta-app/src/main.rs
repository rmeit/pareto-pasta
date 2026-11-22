//! pareto_pasta: edit a die, see its exit-speed map live, and watch the
//! pasta grow in 3D.

mod app;
mod die_canvas;
mod panels;
// The viewport's full API (camera state, extra colour maps) is kept for the
// demo example and future use.
#[allow(dead_code)]
mod viewport;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 920.0]).with_min_inner_size([960.0, 600.0]).with_title("pareto_pasta"),
        ..Default::default()
    };
    let path = std::env::args().nth(1).map(std::path::PathBuf::from);
    eframe::run_native("pareto_pasta", options, Box::new(move |cc| Ok(Box::new(app::App::new(cc, path)))))
}
