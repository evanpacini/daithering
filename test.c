#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#define PGM_MAX_GRAY 255
#define PGM_MAX_GRAY_F 255.0

// PGM image format
typedef struct {
  uint32_t width_;
  uint32_t height_;
  uint16_t max_gray_;
  uint8_t *data_;
} PgmImage;

/**
 * Allocate memory for a PGM image.
 *
 * @param width     The width of the image.
 * @param height    The height of the image.
 * @return          A pointer to the PgmImage, or NULL if an error occurred.
 */
PgmImage *AllocatePgm(uint32_t width, uint32_t height) {
  // Allocate memory for image data
  PgmImage *image = (PgmImage *)calloc(1, sizeof(PgmImage));
  if (!image) {
    fprintf(stderr, "Error: out of memory\n");
    return NULL;
  }
  image->width_    = width;
  image->height_   = height;
  image->max_gray_ = PGM_MAX_GRAY;
  image->data_     = (uint8_t *)calloc(1, width * height * sizeof(uint8_t));
  if (!image->data_) {
    fprintf(stderr, "Error: out of memory\n");
    free(image);
    return NULL;
  }

  return image;
}

/**
 * Read a PGM image from a file.
 *
 * @param filename  The name of the file to read.
 * @return          A pointer to the image data, or NULL if an error occurred.
 */
PgmImage *ReadPgm(const char *filename) {
  // Open file for reading
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error: could not open file '%s'\n", filename);
    return NULL;
  }

  // Read header (magic number, width, height, and max gray value)
  char magic[3];
  uint32_t width;
  uint32_t height;
  uint16_t max_gray;
  if (fscanf(fp, "%2s%*[ \t\r\n]%u%*[ \t\r\n]%u%*[ \t\r\n]%hu%*1[ \t\r\n]",
             magic, &width, &height, &max_gray) != 4) {
    fprintf(stderr, "Error: invalid header in file '%s'\n", filename);
    fclose(fp);
    return NULL;
  }

  // Make sure the magic number is "P5" (binary PGM format)
  if (magic[0] != 'P' || magic[1] != '5') {
    fprintf(stderr, "Error: unsupported file format in file '%s'\n", filename);
    fclose(fp);
    return NULL;
  }

  // Make sure the max gray value is PGM_MAX_GRAY
  if (max_gray != PGM_MAX_GRAY) {
    fprintf(stderr, "Error: max gray value must be PGM_MAX_GRAY\n");
    fclose(fp);
    return NULL;
  }

  // Allocate memory for image data
  PgmImage *image = AllocatePgm(width, height);

  // Read pixel data
  if (fread(image->data_, sizeof(uint8_t), width * height, fp) !=
      width * height) {
    fprintf(stderr, "Error: could not read pixel data from file '%s'\n",
            filename);
    free(image->data_);
    free(image);
    fclose(fp);
    return NULL;
  }

  fclose(fp);
  return image;
}

/**
 * Write a PGM image to a file
 *
 * @param filename  Name of file to write to
 * @param image     Image to write
 * @return          True if successful, false otherwise
 */
bool WritePgm(const char *filename, const PgmImage *image) {
  // Open file for writing
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Error: could not open file '%s' for writing\n", filename);
    return false;
  }

  // Write header (magic number, width, height, and max gray value)
  if (fprintf(fp, "P5\n%u\n%u\n%hu\n", image->width_, image->height_,
              image->max_gray_) < 0) {
    fprintf(stderr, "Error: could not write header to file '%s'\n", filename);
    fclose(fp);
    return false;
  }

  // Write pixel data
  if (fwrite(image->data_, sizeof(uint8_t), image->width_ * image->height_,
             fp) != image->width_ * image->height_) {
    fprintf(stderr, "Error: could not write pixel data to file '%s'\n",
            filename);
    fclose(fp);
    return false;
  }

  fclose(fp);
  return true;
}
