#![allow(clippy::all)]

use std::time::Instant;

use compute_engine::{BaseEngine, ComputeEngine};
use image::{ImageBuffer, Rgba};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyImageToBufferInfo},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

mod shader {
    vulkano_shaders::shader! {ty: "compute", path: "src/shader.comp"}
}

#[cfg(test)]
mod tests;

pub fn entrypoint() {
    // Prepare Engine
    let compute_engine = ComputeEngine::new();

    // Print some info
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Prepare Image
    let image = StorageImage::new(
        compute_engine.get_logical_device().get_device(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(compute_engine.get_logical_device().get_queue_family_index()),
    )
    .unwrap();
    let image_view = ImageView::new_default(image.clone()).unwrap();

    // Prepare Output Buffer
    let output_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    // Prepare Shader
    let shader = shader::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create shader module");

    // Prepare Compute Pipeline
    let compute_pipeline = ComputePipeline::new(
        compute_engine.get_logical_device().get_device(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("failed to create compute pipeline");

    // Prepare Descriptor Set
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, image_view.clone())],
    )
    .expect("failed to create descriptor set");

    // Submit Command Buffer for Computation
    compute_engine.compute(&|engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            engine.get_logical_device().get_device(),
            engine.get_logical_device().get_queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch([1024 / 8, 1024 / 8, 1])
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                output_buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    });

    // Save results
    #[cfg(debug_assertions)]
    let start = Instant::now();

    let buffer_content = output_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("mandelbrot.png").unwrap();

    #[cfg(debug_assertions)]
    let end = Instant::now();

    log::info!("Successfully saved image");

    #[cfg(debug_assertions)]
    log::debug!(
        "Storing image took: {}ms",
        end.duration_since(start).as_millis()
    );
}
