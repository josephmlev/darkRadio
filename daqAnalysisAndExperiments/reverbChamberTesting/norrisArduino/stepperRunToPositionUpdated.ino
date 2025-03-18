#include <AccelStepper.h>
#include <elapsedMillis.h>

// Motor Connections (constant current, step/direction bipolar motor driver)
const int dirPin = 4;
const int stepPin = 5;
const int SW = 3;
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

  pinMode(SW, OUTPUT);
  pinMode(encoderPin, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);

  // Indicator LED shows that program has started
  for (int i = 0; i < 5; i++){
          digitalWrite(LED_BUILTIN, HIGH); // LED is pin 6
          delay(100);
          digitalWrite(LED_BUILTIN, LOW);
          delay(100);
      }
}

void loop() {

  encoderSignal = digitalRead(encoderPin);

  if(Serial.available() > 0){
    String command = Serial.readString();

    // checks for command containing "pos" and no motion (relay off) --> prints current position
    if(command.indexOf('pos') > 0 and digitalRead(SW) == LOW){                     
        Serial.println("Current stepper position is " + String(myStepper.currentPosition()));
    }

    // checks for command containing "pos" and motion (relay on) --> prints target position
    else if(command.indexOf('pos') > 0 and digitalRead(SW) == HIGH){
        Serial.println("Target stepper position is " + String(myStepper.targetPosition()));
    }

    // checks for command containing "step" and no motion (relay off) --> steps forward by 1
    else if (command.indexOf('step') > 0 and digitalRead(SW) == LOW){
        Serial.println("Taking one step forward...");

        desiredPosition = (myStepper.currentPosition() + 1) % stepsPerRev;

        myStepper.moveTo(desiredPosition);

        Serial.println("Done. Moved forward by 1 (micro)step. ");
    }

    // checks for command containing desired step position (number), if found it enables the relay, moves stepper to that position, then disables relay
    else if(isDigit(command[0])){              
                                              
      Serial.println("Step command received, enabling relay... ");
      digitalWrite(SW, HIGH);
      Serial.println("Done. Relay enabled.");

      Serial.println("Initializing motion, please step back from blades... ");
      //digitalWrite(LED, HIGH);
      
      desiredPosition = command.toInt();
      desiredPosition = desiredPosition % stepsPerRev; // accounts for step positions above the max step position

      // indicator LED shows that stepping is starting
      for (int i = 0; i < 5; i++){
          digitalWrite(LED_BUILTIN, HIGH); // LED_BUILTIN is "L" light on board
          delay(100);
          digitalWrite(LED_BUILTIN, LOW);
          delay(100);
      }
      myStepper.moveTo(desiredPosition);  

      Serial.println("Done. Stepping complete, now disabling relay... ");
      digitalWrite(SW, LOW);
      Serial.println("Done. Relay disabled. ");

    }
    else if (command.indexOf("quit") > 0){
      Serial.println("Quit command received. Stopping motion... ");

      myStepper.disableOutputs();           // Disable motor output immediately
      myStepper.setCurrentPosition(myStepper.currentPosition());  // Reset position to current instantly

      digitalWrite(SW, LOW);

      Serial.println("Stirrer successfully stopped at position " + String(myStepper.currentPosition()) + " and relay disabled! ");
    }

    else { // input not recognized
        Serial.println("Input not recognized, plz try again. ");
        for (int i = 0; i < 5; i++){
          digitalWrite(LED, HIGH); // LED is pin 6
          delay(100);
          digitalWrite(LED, LOW);
          delay(100);
        }
    }
  }
  if (encoderSignal == LOW){
      Serial.println("Stirrer has made a full rotation! Stopping motion now...");

      myStepper.disableOutputs();           // Disable motor output immediately
      myStepper.setCurrentPosition(myStepper.currentPosition());  // Reset position to current instantly

      Serial.println("Stirrer successfully stopped! Now setting this to 0 position...");

      myStepper.setCurrentPosition(0);

      Serial.println("Done. This is not position 0.");

  }
  //if (myStepper.distanceToGo() != 0){
  myStepper.run();
  //}
}