Benchmarks
==============

Compare accuracy, speed, and memory on reproducible workloads. Start with
the performance report for practical trade-offs, or choose a study below to
see how the methods behave on a particular kind of data.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Performance and memory
      :link: performance
      :link-type: doc
      :img-top: /auto_examples/02_resize/images/thumb/sphx_glr_07_benchmarking_plot_thumb.png
      :img-alt: Quality-versus-scale curves from the resize benchmark.
      :class-card: example-preview

      Compare complete workflows, reusable plans, native acceleration,
      and memory use. Includes workloads where another method is faster.

   .. grid-item-card:: Controlled 3-D fields
      :link: wavefield3d-study
      :link-type: doc
      :img-top: /_static/wavefield3d-study-accuracy.png
      :img-alt: Accuracy comparison against analytical three-dimensional fields.
      :class-card: example-preview

      Measure resampling error against an analytical target, including
      the stronger polyphase reference added after the original study.

   .. grid-item-card:: Microscopy volumes
      :link: cells3d-study
      :link-type: doc
      :img-top: /_static/cells3d-study-montage.png
      :img-alt: Image slices comparing microscopy downsampling methods.
      :class-card: example-preview

      Inspect a two-channel volume and compare visual detail, accuracy
      metrics, and local runtime.

   .. grid-item-card:: Segmentation validation
      :link: bbbc050-study
      :link-type: doc
      :img-top: /_static/bbbc050-study-example.png
      :img-alt: Example images from the labelled segmentation study.
      :class-card: example-preview

      Check downstream results against manual labels. This study did
      not meet its predeclared quality-superiority criterion.

Microvessel case study
--------------------------

The :doc:`selma3d-vessels-study` compares named downsampling pipelines on
18 held-out labelled 3-D patches. It reports a specific quality-and-speed
result for voxel ranking; it does not establish segmentation superiority.
Use the :doc:`selma3d-demo` to explore one patch interactively in napari.

Reproduce and interpret the results
---------------------------------------

Each report records its data, geometry, methods, environment, and limitations.
Timings are machine-dependent. Coordinate and boundary conventions must match
before an output difference can be interpreted as numerical error.

For a concise record of what the studies support, see :doc:`claims`.
For numerical API guarantees and tested configurations, see
:doc:`project-status` and :doc:`backend-support`.

.. toctree::
   :hidden:
   :maxdepth: 1

   performance
   wavefield3d-study
   cells3d-study
   bbbc050-study
   selma3d-vessels-study
   selma3d-demo
   claims
