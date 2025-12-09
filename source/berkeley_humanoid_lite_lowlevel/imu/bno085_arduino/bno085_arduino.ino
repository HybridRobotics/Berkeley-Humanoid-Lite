// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include <Wire.h>
#include <Arduino.h>
#include <Adafruit_BNO08x.h>


#define DEBUG_MODE        0

#define UART_BAUDRATE     1000000

#define BNO085_ADDRESS    0x4B


typedef struct {
  uint8_t sync_1;
  uint8_t sync_2;
  uint16_t size;
  
  float rw;
  float rx;
  float ry;
  float rz;

  float vx;
  float vy;
  float vz;
} IMUReadings;

IMUReadings readings;

Adafruit_BNO08x bno08x(-1);

sh2_SensorValue_t sensor_value;

uint8_t gyro_updated = 0;
uint8_t quat_updated = 0;


void setReports() {
  /*
  Reading `SH2_GYRO_INTEGRATED_RV` and `SH2_GYROSCOPE_CALIBRATED` ensures around 200 ~ 500 Hz update rate
  */

  // target 200 Hz update rate
  uint32_t report_interval = 5000;

  // Top frequency is reported to be 1000Hz (but freq is somewhat variable)
  if (!bno08x.enableReport(SH2_GYRO_INTEGRATED_RV, report_interval)) {
    Serial.println("[ERROR] Could not enable stabilized remote vector");
  }
  // Top frequency is about 250Hz but this report is more accurate
  // if (!bno08x.enableReport(SH2_ARVR_STABILIZED_RV, report_interval)) {
  //   Serial.println("[ERROR] Could not enable stabilized remote vector");
  // }

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, report_interval)) {
    Serial.println("[ERROR] Could not enable gyroscope");
  }
  // if (!bno08x.enableReport(SH2_GRAVITY, report_interval)) {
  //   Serial.println("[ERROR] Could not enable gravity vector");
  // }
}

void setup(void) {
  Serial.begin(UART_BAUDRATE);

  // while (!Serial) {
  //   delay(10);
  // }
  
  // wait for the chip to bootup
  delay(200);

  Wire.setClock(400000);

  if (!bno08x.begin_I2C(BNO085_ADDRESS)) {
    Serial.println("[ERROR] Failed to find BNO08x chip");
    while (1) {delay(10);}
  }


  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);
  
  // status LED
  pinMode(3, OUTPUT);

  setReports();

  delay(200);

  readings.sync_1 = 0x75;
  readings.sync_2 = 0x65;
  readings.size = 28;
}

void loop() {
  if (bno08x.wasReset()) {
    // Serial.print("[ERROR] sensor was reset ");
    setReports();
  }
  
  if (bno08x.getSensorEvent(&sensor_value)) {
    switch (sensor_value.sensorId) {
      // case SH2_ARVR_STABILIZED_RV:
      //   readings.rw = sensor_value.un.arvrStabilizedRV.real;
      //   readings.rx = sensor_value.un.arvrStabilizedRV.i;
      //   readings.ry = sensor_value.un.arvrStabilizedRV.j;
      //   readings.rz = sensor_value.un.arvrStabilizedRV.k;
      //   break;
      
      case SH2_GYRO_INTEGRATED_RV:
        // faster (more noise?)
        // coordinate transform from Y-forward (w, x, y, z) to X-forward (w_, x_, y_, z_) = (w, y, -x, z)
        readings.rw =  sensor_value.un.gyroIntegratedRV.real;
        readings.rx =  sensor_value.un.gyroIntegratedRV.j;
        readings.ry = -sensor_value.un.gyroIntegratedRV.i;
        readings.rz =  sensor_value.un.gyroIntegratedRV.k;
        // digitalWrite(10, HIGH);
        quat_updated = 1;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        // coordinate transform from Y-forward (rx, ry, rz) to X-forward (rx_, ry_, rz_) = (ry, -rx, rz)
        readings.vx =  sensor_value.un.gyroscope.y;
        readings.vy = -sensor_value.un.gyroscope.x;
        readings.vz =  sensor_value.un.gyroscope.z;
        // digitalWrite(11, HIGH);
        gyro_updated = 1;
        break;

      // case SH2_GRAVITY:
      //   readings.gx = sensor_value.un.gravity.x;
      //   readings.gy = sensor_value.un.gravity.y;
      //   readings.gz = sensor_value.un.gravity.z;
      //   break;
    }
  }

  if (gyro_updated && quat_updated) {
    digitalWrite(3, HIGH);
    Serial.write((uint8_t *)&readings, sizeof(readings));
    gyro_updated = 0;
    quat_updated = 0;
    // digitalWrite(10, LOW);
    // digitalWrite(11, LOW);
    digitalWrite(3, LOW);
  }

}
