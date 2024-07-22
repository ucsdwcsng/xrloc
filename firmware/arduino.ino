#include "esp_camera.h"

typedef struct {
    uint16_t gainceiling;
    uint16_t exposure;
    uint16_t awb;
    uint16_t agc;
    uint16_t aec;
    uint16_t hmirror;
    uint16_t vflip;
    uint16_t brightness;
    uint16_t contrast;
    uint16_t saturation;
    uint16_t sharpness;
    uint16_t denoise;
    uint16_t special_effect;
    uint16_t wb_mode;
    uint16_t ae_level;
    uint16_t aec2;
    uint16_t dcw;
    uint16_t bpc;
    uint16_t wpc;
    uint16_t raw_gma;
    uint16_t lenc;
    uint16_t xmirror;
    uint16_t agc_gain;
    uint16_t aec_value;
    uint16_t aec2_value;
    uint16_t gainceiling_value;
    uint16_t bpc_value;
    uint16_t wpc_value;
    uint16_t raw_gma_value;
    uint16_t lenc_value;
} sensor_status_t;

int led0 = 13;

int val = -1;
int count = 0;
int flag_firsttime = 1;

void setup() {
  // put your setup code here, to run once:
  pinMode(led0, OUTPUT);
  digitalWrite(led0, LOW);
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);  
  digitalWrite(7, LOW);
  digitalWrite(8, LOW);
  digitalWrite(led0, LOW);
  flag_firsttime = 1;
}

void loop() {
  // put your main code here, to run repeatedly:

  sensor_t *sensor = esp_camera_sensor_get();
  sensor_status_t status;
  sensor->get_status(sensor, &status);

  // Estimate the current exposure level based on the current exposure time and analog gain values
  int exposure = (status.shutter / 1000) * status.gainceiling / 16;
  Serial.printf("Exposure: %d\n", exposure);

  if(flag_firsttime == 1){
    delay(1000);
    digitalWrite(led0, LOW);
    delay(1000);
    digitalWrite(led0, HIGH);
    flag_firsttime = 0;
    digitalWrite(7, LOW);
    delay(30000);
    digitalWrite(7, HIGH);
    delay(1000);
    digitalWrite(8, HIGH);
    delay(1000);
    digitalWrite(8, LOW);
  }

  digitalWrite(led0, HIGH);


// Delay/wait for a second (1000 milliseconds = 1 second)
  delay(1000);

// Turn the LED off (LOW is “off” in the Arduino language)
  digitalWrite(led0, LOW);
// Delay/wait for a second
  delay(1000);
  count = count + 1;
}
