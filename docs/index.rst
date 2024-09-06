SplineOps - Spline Operations
=============================

**SplineOps** is a Python-based open-source software library aimed at providing efficient signal processing tools using splines. 
Currently adapting and building on the legacy algorithms developed by the `Biomedical Imaging Group at EPFL <https://bigwww.epfl.ch/>`_ (Lausanne, Switzerland), 
SplineOps is in active development and evolving to support modern computational demands.

With a focus on handling large datasets, SplineOps supports both CPU and GPU computations, offering tools for data smoothing, interpolation, 
and other applications. While still a work in progress, the library aims to integrate with the PyData ecosystem and support a variety of research and engineering needs.

By leveraging modern computing architectures, SplineOps seeks to enhance computational workflows while maintaining the rigor and reliability of the original algorithms.

.. figure:: _static/waveletbird_full.jpeg
   :alt: Main Feature of SplineOps
   :align: center
   :scale: 40%

   Different representations of spline functions and their derivatives

Key Features & Capabilities
===========================

- **Optimized Performance**: Leveraging CPU and GPU architectures to handle large-scale signal data sets effectively.

- **Precision and Flexibility**: High-degree spline interpolations across multiple dimensions.

- **Scalability and Extensibility**: To incorporate new functionalities tailored to specific applications.

.. figure:: _static/feature_01.jpg
   :alt: Key Feature Illustration
   :align: center
   :scale: 60%

   General B-Spline formula

Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation/index
   gpu-support/index
   auto_examples/index
   api/index