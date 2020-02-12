#include <jinterc_util.h>

/**
 * Debugging routines, to be used in GDB, etc.
 */

const char *cepFileName = "floatDump.txt";

void cepName(const char *newName) { cepFileName = newName; }

void cepFloats(void *p, int nbytes) {
    FILE *fp = fopen(cepFileName, "w");
    float *data = (float *)p;
    for (int i = 0; i < nbytes / 4; i++) {
        fprintf(fp, "%d: %f\n", i, data[i]);
    }
    fclose(fp);
}

void copyTensorDims(TfLiteTensor *tflTensor,
                    tabeq::runtime::Tensor *tbqTensor) {
    if (tflTensor == nullptr || tbqTensor == nullptr) return;
    // -- WIP
    throw JintercException("copyTensorDims not yet implemented");
}
void copyTensorPointer(TfLiteTensor *tflTensor,
                       tabeq::runtime::Tensor *tbqTensor) {
    if (tflTensor == nullptr || tbqTensor == nullptr) return;
    // -- WIP
    throw JintercException("copyTensorDims not yet implemented");
}
