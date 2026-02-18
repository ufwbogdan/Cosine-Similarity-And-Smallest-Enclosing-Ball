Here are 2 lab tasks that show use of vectorization using intel intrinisic functions.

-> Task 1:
  - Given two n-dimensional vectors, A and B, the cosine similarity is a real value in the range [-1, 1] computed as follows:
   ```math
    s(A, B) = \frac{A \cdot B}{\|A\| \, \|B\|}
    = \frac{\sum_{i=1}^{n} a_i b_i}
    {\sqrt{\sum_{i=1}^{n} a_i^2} \, \sqrt{\sum_{i=1}^{n} b_i^2}}
   ```

  - Cosine similarity is a measure used to quantify how similar two real-valued vectors of the same dimension are. In information retrieval, for instance, documents can be represented as vectors whose elements contain scores associated with specific terms from a given
    dictionary. Each score reflects how representative a term is for the corresponding document. Cosine similarity is then used to measure the degree of similarity between two documents.
  - The vectorized version implemented can be found in the vector.hpp file, with it automatically adapting to the compilation target, using compiler defined-macros to select the highest-performance available vectorization extenstion (e.g. AVX, SSE) and falling back to the
    scalar operations if no SIMD extension is available.

-> Task 2:
  - Given an array of d-dimensional points, find an enclosing d-dimensional sphere (not necessarily the smallest).
  - The procedure consists of three main steps:
      1. Sweep the set of points to find the minimum and maximum value on each dimension.
      2. Use those values to outline a d-dimensional box then calculate its center c.
      3. Sweep again the set of points and find the farthest point p from c then calculate the radius ||p - c||.
         
  - In computer graphics, a mesh is a collection of points representing an object. Meshes can be generated algorithmically, obtained by scanning a real object, or manually created by artists.
  - Finding the smallest enclosing ball is a classic problem in computational geometry. It is often used, for example, to accelerate the collision detection phase for objects sharing the same environment: each mesh is first approximated by a tight hypersphere, then these
    hypersphere, then these hyperspheres are tested to detect any pairwise collisions.
  - In the mesh.hpp file you can find the vectorized version of the mesh struct working for any value of D.
    
Both tasks use a makefile for easy access to global variables and compile flags.
