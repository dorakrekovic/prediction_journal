#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "xprintf.h"
#include "timer_interface.h"
#include "hx_drv_scu.h"
#include "hx_drv_iic.h"

#include "model_data_conv1D.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#define ESP_I2C_SLAVE_ADDR 0x42
#define LAG 24  // Must match model
#define BUFFER_SIZE LAG  // We expect 24 readings
#define TRAIN_MEAN 286.537f
#define TRAIN_STD 8.543f

// Tensor arena
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// I2C error callback
volatile uint32_t g_err_cb = 0;
void i2cm_0_err_cb(void *status) {
    HX_DRV_DEV_IIC *iic_obj = status;
    HX_DRV_DEV_IIC_INFO *iic_info_ptr = &(iic_obj->iic_info);
    g_err_cb = 1;
    xprintf("[%s] err:%d \n", __FUNCTION__, iic_info_ptr->err_state);
}

int app_main(void) {
    xprintf("Start + TFLite Prediction\n");

    uint8_t rbuffer[BUFFER_SIZE];
    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_I2C_M_SCL, 1);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_I2C_M_SDA, 1);
    hx_drv_i2cm_init(USE_DW_IIC_0, HX_I2C_HOST_MST_0_BASE, DW_IIC_SPEED_FAST);
    hx_drv_i2cm_set_err_cb(USE_DW_IIC_0, i2cm_0_err_cb);

    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        xprintf("Model schema version mismatch\n");
        return -1;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        xprintf("Tensor allocation failed\n");
        return -1;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    while (1) {
        uint8_t dummy_reg = 0x00;
        IIC_ERR_CODE_E i2c_err = hx_drv_i2cm_write_restart_read(
            USE_DW_IIC_0, ESP_I2C_SLAVE_ADDR,
            &dummy_reg, 1,
            rbuffer, BUFFER_SIZE);

        if (i2c_err != IIC_ERR_OK) {
            xprintf("I2C ERROR: %d\n", i2c_err);
        } else {
            xprintf("Received last %d hourly readings from ESP:\n", BUFFER_SIZE);
            float norm_input[LAG];
            for (int i = 0; i < LAG; i++) {
                float temp_c = (float)rbuffer[i];
                float temp_k = temp_c + 273.15f;
                norm_input[i] = (temp_k - TRAIN_MEAN) / TRAIN_STD;
                xprintf("  [%02d] = %.2f °C\n", i, temp_c);
            }

            // Quantize and feed to model input
            for (int i = 0; i < LAG; i++) {
                input->data.int8[i] = (int8_t)(norm_input[i] / input->params.scale + input->params.zero_point);
            }

            // Run inference
            if (interpreter->Invoke() != kTfLiteOk) {
                xprintf("Inference failed\n");
                continue;
            }

            // Dequantize output
            int8_t q_out = output->data.int8[0];
            float norm_pred = (q_out - output->params.zero_point) * output->params.scale;
            float pred_k = norm_pred * TRAIN_STD + TRAIN_MEAN;
            float pred_c = pred_k - 273.15f;

            float last_temp_c = (float)rbuffer[LAG - 1];

            xprintf("Predicted next temp: %.2f °C\n", pred_c);
            xprintf("Last actual temp:    %.2f °C\n", last_temp_c);

            float diff = fabs(pred_c - last_temp_c);
            if (diff > 2.0f) {
                xprintf("WARNING: Temp deviation exceeds 2°C!\n");
            } else {
                xprintf("Temp prediction within normal range.\n");
            }

            xprintf("------------------------------------------\n");
        }

        hx_drv_timer_cm55x_delay_ms(10000, TIMER_STATE_DC);  //  10 sec
    }

    return 0;
}
