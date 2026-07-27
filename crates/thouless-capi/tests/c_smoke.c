#include "thouless.h"

#include <math.h>
#include <stddef.h>

static int run_model_case(void) {
    double primitive[] = {1.0};
    size_t periodic_axes[] = {0};
    ThoulessF64MatrixView lattice = {primitive, 1, 1, 1, 1};
    ThoulessModelBuilder *builder = NULL;
    if (thouless_model_builder_create(
            lattice, periodic_axes, 1, &builder) != THOULESS_SUCCESS) {
        return 2;
    }

    const uint8_t label[] = {'s'};
    double position[] = {0.0};
    size_t orbital = SIZE_MAX;
    if (thouless_model_builder_add_orbital(
            builder, label, 1, position, 1, 1, &orbital) != THOULESS_SUCCESS) {
        return 3;
    }
    int32_t offset[] = {1};
    ThoulessComplex64 hopping = {-1.0, 0.0};
    if (thouless_model_builder_add_hopping(
            builder, orbital, orbital, offset, 1, hopping) != THOULESS_SUCCESS) {
        return 4;
    }

    ThoulessModel *model = NULL;
    if (thouless_model_builder_build(builder, &model) != THOULESS_SUCCESS) {
        return 5;
    }
    size_t state_count = 0;
    if (thouless_model_state_count(model, &state_count) != THOULESS_SUCCESS ||
        state_count != 1) {
        return 6;
    }

    double momentum[] = {0.0};
    double eigenvalue[] = {NAN};
    ThoulessComplex64 eigenvector[] = {{0.0, 0.0}};
    ThoulessC64MatrixMut vectors = {eigenvector, 1, 1, 1, 1};
    if (thouless_model_eigensystem(
            model, momentum, 1, eigenvalue, 1, vectors) != THOULESS_SUCCESS) {
        return 7;
    }
    if (fabs(eigenvalue[0] + 2.0) > 1e-12) {
        return 8;
    }

    if (thouless_model_destroy(model) != THOULESS_SUCCESS ||
        thouless_model_builder_destroy(builder) != THOULESS_SUCCESS) {
        return 9;
    }
    return 0;
}

static int run_ad_case(void) {
    ThoulessComplex64 base_data[] = {
        {-1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}
    };
    ThoulessComplex64 direction_data[] = {
        {0.0, 0.0}, {1.0, 0.0},
        {1.0, 0.0}, {0.0, 0.0}
    };
    ThoulessComplex64 target_data[] = {
        {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}
    };
    ThoulessC64MatrixView base = {base_data, 2, 2, 2, 1};
    ThoulessC64Tensor3View directions = {
        direction_data, 1, 2, 2, 4, 2, 1
    };
    ThoulessC64MatrixView target = {target_data, 2, 2, 2, 1};
    double parameters[] = {0.2};
    double value = NAN;
    double gradient[] = {NAN};
    if (thouless_ad_affine_projector_value_and_grad(
            base,
            directions,
            parameters,
            1,
            1,
            target,
            1e-8,
            &value,
            gradient,
            1) != THOULESS_SUCCESS) {
        return 10;
    }
    double expected_value = 0.5 * (1.0 - 1.0 / sqrt(1.04));
    double expected_gradient = 0.1 / pow(1.04, 1.5);
    if (fabs(value - expected_value) > 1e-12 ||
        fabs(gradient[0] - expected_gradient) > 1e-12) {
        return 11;
    }
    return 0;
}

int main(void) {
    if (thouless_abi_version() != THOULESS_ABI_VERSION) {
        return 1;
    }
    for (size_t iteration = 0; iteration < 100; ++iteration) {
        int status = run_model_case();
        if (status != 0) {
            return status;
        }
    }
    return run_ad_case();
}
