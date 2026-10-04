# CL-JPEG
Baseline JPEG codec written in Common Lisp.

This particular version is fork of [GitHub](https://github.com/sharplispers/cl-jpeg), written by Eugene Zaikonnikov, 1999-2017. Valuable contributors, in no particular order: Henry Harrington, Kenan Bölükbaşı, Manuel Giraud, Cyrus Harmon and William Halliburton.

## Image Format

* B, G, R pixels in case of colorspace-converted three component image
* Y, Cb, Cr pixels in case where colorspace conversion of three component image was disabled
* array of grayscale pixels in case of single component
* array of 2 or 4 pixels in the case of two or four component image respectively

This particular version is my personal playground to play with SIMD, AVX-512 and LLMs with SBCL.

Currently SIMD acceleration includes:

* **AVX2 Forward DCT (Encoder)**: Accelerated 8x8 2D LLM-DCT using register transposes. Delivers 5.13x DCT kernel speedup (2.80 Gpix/s) and provides 100% bit-for-bit mathematical equivalence with scalar CL-JPEG output.

* **AVX2 / AVX-512 Inverse DCT (Decoder)**: Accelerated 8x8 2D inverse LLM-DCT with dequantization. Produces 100% bit-for-bit identical output to scalar `inverse-llm-dct` across all inputs (including signed 16-bit intermediate wrap preservation). Delivers 2.9x–3.8x in-situ kernel speedup and 1.16x–1.17x end-to-end decode speedup on natural images (1.38x–1.47x full SIMD pipeline).

* **AVX-512 Inverse Color Conversion (Decoder)**: Accelerated YCbCr -> RGB conversion for AMD Zen 5 and Intel AVX-512 processors (AVX512F, AVX512BW, AVX512VBMI), processing 16 pixels per chunk with store-forwarding avoidance.

* **Fused Planar SIMD Pipeline (Decoder)**: Fused native quarter-resolution chroma decode with 2-row AVX-512BW+VBMI color conversion, completely eliminating the separate upsample replication pass. This is a speed-for-memory trade-off: delivering 3%–9% faster end-to-end decode latency in exchange for 1.5× retained intermediate memory (4.5·W·H vs 3.0·W·H in interleaved SIMD) and 10% higher single-shot allocation (4.21 MB vs 3.81 MB on 512²). Intermediate plane allocation overhead is 0 bytes when image descriptors are reused. On memory, interleaved SIMD wins on both single-shot and reused-descriptor footprints.

### IDCT Sparsity & End-to-End Speedup

Direct ablation measurements confirm that IDCT accounts for 14%–20% of natural image decode runtime. Empirical profiling reveals that **>98.5% of real blocks are Pure DC (10–52%) or Sparse AC (46–89%)**. The scalar baseline takes effective early-outs on zero columns, which closes much of the potential gap with SIMD. In-situ ablation (stubbing IDCT) demonstrates that SIMD IDCT achieves a **2.9x–3.8x in-situ kernel speedup**, translating directly to the measured **1.16x–1.17x end-to-end decode speedup** (and **1.38x–1.47x** with SIMD color conversion), in close agreement with Amdahl's Law.


SIMD acceleration is enabled automatically when supported hardware is detected at runtime. It can be controlled via the `*use-simd*` special variable:
* `:auto` (default) - Use AVX-512 kernels when supported by hardware and runtime.
* `:on`   - Force SIMD kernels.
* `:off`  - Disable SIMD and force scalar execution.

### Breaking Change: Chroma Bias & Bit Exactness Notice

In SIMD version, the decoder lookup tables (`+cr-r-tab+`, `+cb-g-tab+`, `+cr-g-tab+`, `+cb-b-tab+`) were updated to use standard JFIF chroma zero bias 128 instead of legacy 127.

- Decoded output differs from previous versions (<= 2.8.x) by <= 1 LSB per channel. Callers performing round-trip pixel-exact assertions against older CL-JPEG output should be aware that output now strictly adheres to the JFIF specification.

- Because the scalar path uses 54-bit fixed point (shift = 54) and the SIMD path uses 15-bit `vpmulhrsw`, the two paths differ by +/- 1 LSB on approximately 8% of pixels (where two-step vs one-step rounding straddles a rounding boundary). Both paths are JFIF-compliant. Do not attempt to eliminate this difference; doing so would require either slower 32-bit SIMD arithmetic or matching the scalar rounding exactly (not expressible with `vpmulhrsw`).

- SIMD acceleration requires the fixed scalar tables to agree within +/- 1 LSB.

The rest below is the original README.md from the original source:

## Decoding

### `(DECODE-IMAGE FILENAME &KEY BUFFER (COLORSPACE-CONVERSION T) CACHED-SOURCE-P)`
* FILENAME - jpeg file name
* BUFFER - optional reusable storage for output image
* COLORSPACE-CONVERSION controls if YUV to RGB should be performed
* CACHED-SOURCE-P - the file input layer behaviour

Returns (multiple-valued) decoded IMAGE array in the same format as encoder source image, image HEIGHT and image WIDTH.

### `(DECODE-STREAM STREAM &KEY BUFFER (COLORSPACE-CONVERSION T) DESCRIPTOR CACHED-SOURCE-P)`
* STREAM - jpeg image stream or NIL in case where the buffer is pre-allocated in DESCRIPTOR
* BUFFER - optional reusable storage for output image
* COLORSPACE-CONVERSION controls if YUV to RGB should be performed
* DESCRIPTOR is a JPEG:DESCRIPTOR structure. Can be reused for performance reasons bulk processing applications
* CACHED-SOURCE-P - the file input layer behaviour

Returns (multiple-valued) decoded IMAGE array in the same format as encoder source image, image HEIGHT and image WIDTH.

### `(ALLOCATE-BUFFER HEIGHT WIDTH NCOMP)`
* HEIGHT - image height
* WIDTH - image width
* NCOMP - number of image components

Allocates and retruns a buffer suitable for storage of the output image in the specified geometry.

### `(JPEG-FILE-DIMENSIONS FILENAME)`
* FILENAME - the file to be probed

Returns image height, width, number of components, and the type of Adobe colorpsace transform encoded in the image (:YCBCR-RGB, :YCCK-CMYK).

### `(CONVERT-CMYK-TO-RGB BUFFER H W &key RGB-BUFFER)`
* BUFFER - the input CMYK pixel buffer
* H,W - image dimensions
* RGB-BUFFER - optional reusable storage for output RGB image

Returns an RGB conversion of the supplied CMYK image.

###`(JPEG-TO-BMP &key INFILE OUTFILE)`

Converts JPEG file to BMP file.

## Encoding

### `(ENCODE-IMAGE FILENAME IMAGE NCOMP H W &key SAMPLING Q-TABS Q-FACTOR)`

* FILENAME - output file name
* IMAGE - input array of pixels
* NCOMP - number of components in the image
* H, W - image dimensions
* SAMPLING - sampling frequency for ncomp components by X and Y axis, e.g. '((2 2) (1 1) (1 1)) for three components, can be omitted for grayscale and RGB images
* Q-TABS - specifies quantization tables vector, should be 1 for 1, 2 for 2, 2 for 3 and 4 entries for 4 components
* Q-FACTOR - quality specifier (1-64), default is 64

Encodes the supplied image using the sampling specifier, quantization tables and quantization factor.

### `(ENCODING-WRAPPER FILENAME IMAGE NCOMP H W &key (QUALITY 4))`

* FILENAME - output file name
* IMAGE - input array of pixels
* NCOMP - number of components in the image
* H, W - image dimensions
* QUALITY - quality factor on scale from 1 to 5

## TODO

* Add progressive JPEG support in decoder
