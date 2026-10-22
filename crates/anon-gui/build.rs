fn main() {
    let config = slint_build::CompilerConfiguration::new()
        .with_style("fluent-dark".to_string())
        .with_bundled_translations("lang")
        .with_default_translation_context(slint_build::DefaultTranslationContext::None);
    println!("cargo:rerun-if-changed=lang");
    slint_build::compile_with_config("ui/review.slint", config).expect("slint compile");
}
