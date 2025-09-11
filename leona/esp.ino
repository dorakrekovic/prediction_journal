#include <Wire.h>
#include "DHT.h"

#define DHTPIN 5
#define DHTTYPE DHT11
#define I2C_SLAVE_ADDR 0x42
#define LAG 24

DHT dht(DHTPIN, DHTTYPE);
float temp_sum = 0;
int temp_count = 0;
unsigned long last_hour_check = 0;

uint8_t tempBuffer[LAG];
uint8_t bufferHead = 0;
bool bufferFilled = false;

void requestEvent() {
  Wire.write(tempBuffer, LAG);  // Send 24 bytes (°C as uint8)
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  Wire.begin(I2C_SLAVE_ADDR);
  Wire.onRequest(requestEvent);
  memset(tempBuffer, 0, LAG);
  last_hour_check = millis();
}

void loop() {
  float t = dht.readTemperature();  // °C

  if (!isnan(t)) {
    temp_sum += t;
    temp_count++;
  }

  if (millis() - last_hour_check >= 3600000UL) { // 1 hour
    float avg_temp = temp_sum / max(temp_count, 1);
    Serial.print("Hourly avg temp: ");
    Serial.println(avg_temp);

    // Store in buffer
    tempBuffer[bufferHead] = (uint8_t)avg_temp;
    bufferHead = (bufferHead + 1) % LAG;
    if (bufferHead == 0) bufferFilled = true;

    // Reset for next hour
    temp_sum = 0;
    temp_count = 0;
    last_hour_check = millis();
  }

  delay(60000);  // read every 1 min
}
