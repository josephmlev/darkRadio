// Include the AccelStepper Library
#include <AccelStepper.h>
#include <elapsedMillis.h>

// Motor Connections (constant current, step/direction bipolar motor driver)
const int dirPin = 4;
const int stepPin = 5;
const int swPin = 3;
const int encoderPin = 2;

const int LED = 6;

int microstepNum = 4;
long gearRatio = 3L;

long degPerStep = .9/microstepNum;
int stepsPerRev = 360/degPerStep*gearRatio;

long maxStepRate = 25L; // steps / second

long stepRate = 20L; // steps / second

long numRev = 1L;

int numSteps = int(stepsPerRev * numRev);

AccelStepper myStepper(AccelStepper::DRIVER, stepPin, dirPin);           // works for a4988 (Bipolar, constant current, step/direction driver)

elapsedMillis printTime;    // one second info printout timer.

bool relayON = false;

int encoderSignal = 1;

int desiredPosition = 0;

void setup() {
  Serial.begin(9600);
  myStepper.setMaxSpeed(200);    // must be equal to or greater than desired speed.
  myStepper.setAcceleration(10);
  myStepper.setCurrentPosition(0);

  pinMode(swPin, OUTPUT);
  pinMode(encoderPin, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  Serial.println("hi4");
  encoderSignal = digitalRead(encoderPin);

  if(Serial.available() > 0){
    String command = Serial.readString();


    if (command == "relayON"){ // turns on relay which enables power inside RC
      Serial.println("Command = Relay ON");
      relayON = true;
      Serial.println("Done.");
      //digitalWrite(LED, HIGH);
      //delay(5000);
      //digitalWrite(LED, LOW);
    }

    else if (command == "relayOFF"){ // turns off relay which disables power inside RC
      Serial.println("Command = Relay OFF");
      relayON = false;
      Serial.println("Done.");
      //digitalWrite(LED, HIGH);
      //delay(5000);
      //digitalWrite(LED, LOW);
    }




    else if (isDigit(command[0]) and relayON == true){ // turns on stepping
      Serial.println("Beginning steps...");
      //digitalWrite(LED, HIGH);
      
      desiredPosition = command.toInt();

      desiredPosition = desiredPosition % stepsPerRev;

      myStepper.moveTo(desiredPosition);

      Serial.println("Stepping ON");

      for (int i = 0; i < 5; i++){
          digitalWrite(LED, HIGH); // LED is pin 6
          delay(100);
          digitalWrite(LED, LOW);
          delay(100);
      }
      Serial.println("Stepping done.");
      //delay(100);
      //digitalWrite(LED, LOW);
    }

    else { // input not recognized
        Serial.println("Input not recognized, plz try again");
        for (int i = 0; i < 5; i++){
          digitalWrite(LED, HIGH); // LED is pin 6
          delay(100);
          digitalWrite(LED, LOW);
          delay(100);
        }
    }
  }
  if (relayON == true){
    digitalWrite(swPin, HIGH);
  }

  if (relayON == false){
    digitalWrite(swPin, LOW);
  }
  
  if (encoderSignal == LOW){
      Serial.println("Stirrer has made a full rotation! Stopping motion now...");
      //digitalWrite(LED, HIGH);

      myStepper.disableOutputs();           // Disable motor output immediately
      myStepper.setCurrentPosition(myStepper.currentPosition());  // Reset position to current instantly
      //relayON = false // try this out instead of setting speed to 0

      Serial.println("Stirrer successfully stopped!");
    
      //delay(100);
      //digitalWrite(LED, LOW); 
  }
  //if (myStepper.distanceToGo() != 0){
  myStepper.run();
  //}
}