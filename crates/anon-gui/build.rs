fn main() {
    let config = slint_build::CompilerConfiguration::new().with_style("fluent-dark".to_string());
    slint_build::compile_with_config("ui/review.slint", config).expect("slint compile");
}
