#include "Fsm.h"

#define SPDT_PIN 24
#define TEMP_PIN 13
#define THERMISTOR_PIN 0

float voltage, temperature;
float offset = 3477;
float refTemp = 298.15;

void on_spdt_on_enter() 
{
    Serial.println("Entering ON state");
}

void on_spdt_on_exit() 
{
    Serial.println("Exiting ON state");
}

void on_trans_spdt_on_spdt_off()
{
  Serial.println("Transition from ON to OFF");
  digitalWrite(SPDT_PIN, 0);
}

void on_spdt_off_enter() 
{
    Serial.println("Entering OFF state");
}

void on_spdt_off_exit() 
{
    Serial.println("Exiting OFF state");
}

void on_trans_spdt_off_spdt_on()
{
  Serial.println("Transition from OFF to ON");
  digitalWrite(SPDT_PIN, 1);
}

void on_temp_meas_enter()
{
  digitalWrite(TEMP_PIN, HIGH);
  voltage = analogRead(THERMISTOR_PIN) + 2;
  digitalWrite(TEMP_PIN, LOW);
  // +2 is to give same temp as Joseph's meat thermometer
  temperature = 1 / ((1/offset) * log(1/(float(voltage)/1023) - 1) + 1/refTemp) - 273.15 + 2;
  Serial.print("TEMP: ");
  Serial.println(temperature);
}

State state_spdt_on(&on_spdt_on_enter, NULL, &on_spdt_on_exit);
State state_spdt_off(&on_spdt_off_enter, NULL, &on_spdt_off_exit);

State state_temp_meas(&on_temp_meas_enter, NULL, NULL);

Fsm fsm_spdt(&state_spdt_off);
Fsm fsm_temp(&state_temp_meas);

byte incomingByte = '0';

void setup() 
{
  Serial.begin(115200);
  analogReference(EXTERNAL);

  pinMode(TEMP_PIN, OUTPUT);
  pinMode(SPDT_PIN, OUTPUT);
  //pinMode(TEMP_PIN, OUTPUT);
  fsm_spdt.add_transition(&state_spdt_on, &state_spdt_off,
                   SPDT_PIN,
                   &on_trans_spdt_on_spdt_off);
  fsm_spdt.add_transition(&state_spdt_off, &state_spdt_on,
                   SPDT_PIN + 1,
                   &on_trans_spdt_off_spdt_on);
                 
  fsm_spdt.run_machine();

  fsm_temp.add_transition(&state_temp_meas, &state_temp_meas, TEMP_PIN,
                          NULL);

  fsm_temp.run_machine();
}


void loop() 
{
  if(Serial.available() > 0)
  {
    incomingByte = Serial.read();
  }
  if(incomingByte == '0')
  {
    Serial.println("TRANSITION 0");
    fsm_spdt.trigger(SPDT_PIN);
  
  }
  else if(incomingByte == '1')
  {
     Serial.println("TRANSITION 1");
     fsm_spdt.trigger(SPDT_PIN + 1);
  }
  else if(incomingByte == '2')
  {
     Serial.println("TRANSITION 2");
     fsm_temp.trigger(TEMP_PIN);
  }
  incomingByte = 'x';
}
