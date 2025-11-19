#include <ArduinoJson.h>

// --- Sensor Libraries ---
#include <Wire.h>
#include "MAX30100_PulseOximeter.h"
#include <Adafruit_BMP280.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// --- SETTINGS FROM YOUR FRIEND'S CODE ---
#define SDA_PIN 21
#define SCL_PIN 22

// --- Global Objects ---
StaticJsonDocument<1024> jsonDoc;
char jsonBuffer[1024]; 

PulseOximeter pox;
Adafruit_BMP280 bmp;
Adafruit_MPU6050 mpu;

// --- Timing ---
// We will send data at 25Hz (every 40ms)
// to match the SAMPLING_RATE_HZ in app.py
long lastSendTime = 0;
int sendInterval = 40; // 40ms = 25 Hz

void setup() {
  Serial.begin(115200);
  delay(1000);

  // --- Initialize I2C (from friend's code) ---
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000); // standard I²C speed
  delay(500);

  // --- Initialize BMP280 (from friend's code) ---
  Serial.println("Initializing BMP280...");
  if (!bmp.begin(0x76)) Serial.println("BMP280 NOT detected!");
  else Serial.println("BMP280 detected!");

  // --- Initialize MPU6050 (from friend's code) ---
  Serial.println("Initializing MPU6050...");
  if (!mpu.begin()) Serial.println("MPU6050 NOT detected!");
  else {
    Serial.println("MPU6050 detected!");
    // Using your friend's settings
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // --- Initialize MAX30100 (from friend's code) ---
  delay(500); 
  Serial.println("Initializing MAX30100...");
  if (!pox.begin()) {
    Serial.println("MAX30100 NOT detected!");
  } else {
    Serial.println("MAX30100 detected!");
    // Using your friend's setting
    pox.setIRLedCurrent(MAX30100_LED_CURR_7_6MA);
  }
  
  Serial.println("\nUSB sender started. Sending data...");
}

void loop() {
  // --- This is the most important function ---
  // It reads all new data from the sensor
  pox.update();

  // --- Check if it's time to send a new packet ---
  // We pace it manually to 25Hz
  if (millis() - lastSendTime > sendInterval) {
    lastSendTime = millis();
    
    // --- Read other sensors ---
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    float env_temp = bmp.readTemperature();
    float pressure = bmp.readPressure(); // Pa
  
    // --- Build the FULL JSON Object ---
    // We send ALL data your mood_detector.py needs
    jsonDoc.clear();
    jsonDoc["timestamp"] = millis();
    
    // MAX30100 Data
    jsonDoc["hr"] = pox.getHeartRate();
    jsonDoc["spo2"] = pox.getSpO2();
    jsonDoc["ibi"] = pox.getInterBeatIntervalMs(); // CRITICAL for HRV
    jsonDoc["ppg"] = pox.getRawIR(); // CRITICAL for PAV
  
    // MPU6050 Data
    JsonObject mpuData = jsonDoc.createNestedObject("acc");
    mpuData["x"] = a.acceleration.x;
    mpuData["y"] = a.acceleration.y;
    mpuData["z"] = a.acceleration.z;
    
    JsonObject gyroData = jsonDoc.createNestedObject("gyro");
    gyroData["x"] = g.gyro.x;
    gyroData["y"] = g.gyro.y;
    gyroData["z"] = g.gyro.z;
  
    // BMP280 Data
    jsonDoc["temp"] = env_temp;
    jsonDoc["pressure"] = pressure / 100.0F; // Convert to hPa
  
    // --- Send the JSON over Serial (USB) ---
    serializeJson(jsonDoc, jsonBuffer);
    Serial.println(jsonBuffer); // Send the JSON string with a newline
  }
  
  // Small delay to keep the loop healthy
  delay(5); 
}