// Include the AccelStepper Library
#include <AccelStepper.h>
#include <elapsedMillis.h>

// Motor Connections by pin number (constant current, step/direction bipolar motor driver)
const int dirPin = 4;
const int stepPin = 5;
const int swPin = 3;
const int encoderPin = 2;

const int LED = 6;

// change microstepNum to whatever is set on the stepper driver
int microstepNum = 1;
int gearRatio = 3;

long degPerStep = .9;
int stepsPerRev = 360/degPerStep*gearRatio; // 1200 steps/rev without microstepping, multiply by microstep num to get real value

AccelStepper myStepper(AccelStepper::DRIVER, stepPin, dirPin);           // works for a4988 (Bipolar, constant current, step/direction driver)

int encoderSignal = 1;

int desiredPosition = 0;
int stepNum = 0;
int chopPos = 0;

elapsedMillis timeElapsed;    // creating timer for startup

// these bools control position control after finding zero
bool start = false; //
bool zeroFlag = true; //ensures that the initialization code for starting the zero-finding procedure runs only once 
bool stopFlag = false; //used to manage the transition immediately after the encoder (chop) is detected
bool pwrCycleFlag = false; //when finding zero, commands a pwr cycle after stopping at true zero
bool stepCmdFlag = false; // true after receiving a `step` serial command and becomes false after printing

void setup() {
  Serial.begin(115200);
  myStepper.setMaxSpeed(400);    // [steps/sec]
  myStepper.setAcceleration(40); // [steps/sec^2]

  pinMode(swPin, OUTPUT);
  pinMode(encoderPin, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  encoderSignal = digitalRead(encoderPin);
  if (Serial.available() > 0){ // this code runs if data is being sent via serial connection
  /*================================================================================
  Serial Command Descriptions:
    1. "step <num>"
        - Moves the stepper motor by a relative number of steps.
        returns: Serial.println(stepped to pos = <int(current position)>)
    2. "zero"
        - Starts the zero-finding routine to detect the chop (zero) signal.
          - Steps until the chop is detected. 
          - Decelearates to a stop. 
          - Moves to zero. 
          - Power cycles. 
          -  Serial.println("Stopped at A+ zero")
    3. "quit"
        - Disables the power relay, effectively stopping the motor.
        - Closes the serial communication by signaling a shutdown of the system.
        returns: nothing
    4. "power"
        - Turns off the relay powering the stepper motor.
        - The motor can be re-enabled with any command other than 'quit'.
        returns: Serial.println("Power turning off...")
    5. "pc"
        - Executes a power cycle by turning the power off then on again.
        - This action is used to latch the system to the proper A+ zero position.
        returns: Serial.println("Power cycle complete. Power is on")
    6. "getPos"
        - Queries and prints the current position of the stepper motor.
        returns: returns: Serial.println(Current position = <int(current position)>)

    Any unrecognized command causes the system to prompt for re-entry of a valid command.
  ================================================================================*/
    String command = Serial.readString();
    if (command.substring(0, 5) == "step "){ // command should be of form "step #"; causes stepper to step by "#" number of steps; "#" should not be entered in microsteps; stepper will NOT find zero first before moving
      stepNum = command.substring(5).toInt();
      stepNum = stepNum * microstepNum;
      start = false;
      stepCmdFlag = true;
      //relativeStepping = true;
      digitalWrite(swPin, HIGH);
      delay(1000);
      myStepper.move(stepNum); //  will move the stepper in RELATIVE coordinates from (current position) -> (current position + stepNum)
      //timeElapsed = 0;
    }
    else if (command.substring(0, 4) == "zero"){ // makes stepper find zero using chopper (will not move to position after zeroing)
      stepNum = 0;
      start = true;
      digitalWrite(swPin, HIGH);
      delay(500);
      timeElapsed = 0;
    }
    else if(String(command) == "quit"){ // quit command disables power and closes serial connection. 
      digitalWrite(swPin, LOW);
      start = false;
      myStepper.moveTo(myStepper.currentPosition()); // come back to this, might be causing problems (pulses stop briefly then restart showing deceleration when quit ran)
      myStepper.stop();
      Serial.println("Sayonara!");
    }
    else if (command.substring(0, 5) == "power"){ // this is command to turn off relay (disables power to mode stirrer system), this will cause issues if ran when stepper motor is not in an integer position (in between positions)
      digitalWrite(swPin, LOW); // power can be restored by running any of the other commands (except quit) as they enable power by default.
      start = false;
      myStepper.stop();
      Serial.println("Power turning off...");
    }
    else if (command.substring(0, 2) == "pc"){ // This turns off and on the power to find the proper A+ zero
      delay(1000);
      digitalWrite(swPin, LOW);
      delay(1000);
      digitalWrite(swPin, HIGH);
      delay(1000);
      Serial.println("Power cycle complete. Power is on");
    }
    else if (command.substring(0, 6) == "getPos"){ // Query the position
      Serial.println(String("Current position = ") + myStepper.currentPosition());
    }
    else { // input not recognized
      Serial.println("Input not recognized, please try again, sir");
    }
  }
  if(start == true){ // this code runs if the stepper is finding zero, it checks after every step taken to see if a chop is detected
    if(zeroFlag == true){
      //Serial.println("here ");
      myStepper.setCurrentPosition(1); //prevents having to mod 
      myStepper.move(20000);
      zeroFlag = false;
    }
    if (encoderSignal == LOW and timeElapsed > (1 * 1000)){ // check if chop has been found within 5 seconds of startup, this code run if stepper is finding zero only
      chopPos = myStepper.currentPosition();
      myStepper.stop();
      //Serial.println(String("current Position ") + chopPos);
      start = false;
      stopFlag = true;
    }
    else if(encoderSignal == HIGH and timeElapsed > (600000)){ // if 10 mins passes then there is a problem
      myStepper.stop();
      myStepper.setSpeed(0);
      start = false;
      digitalWrite(swPin, LOW);
      Serial.println("Stepper ran for 10 mins and did not detect chop. Check on system.");
    } 
  }
  if(start == false && stopFlag == true && !myStepper.isRunning()) { //it is stopped at chopPos past zero. Command move to true zero
    long stepsPastChop = myStepper.currentPosition() - chopPos;
    delay(1000);
    //Serial.println("moving to true zero ");
    myStepper.move(4804 - stepsPastChop); //4803, 9605
    stopFlag = false;
    zeroFlag = true;
    pwrCycleFlag = true;
  }
  if(start == false && pwrCycleFlag == true && !myStepper.isRunning()) { //we are stopped at true zero. power cycle to latch to an A+ position. Set currentPosition to zero.
    delay(1000);
    digitalWrite(swPin, LOW);
    delay(1000);
    digitalWrite(swPin, HIGH);
    delay(1000);
    digitalWrite(swPin, LOW);
    delay(1000);
    digitalWrite(swPin, HIGH);
    delay(1000);
    Serial.println("Stopped at A+ zero");
    myStepper.setCurrentPosition(0);
    pwrCycleFlag = false;
  }
  if(stepCmdFlag == true && !myStepper.isRunning()){
    stepCmdFlag = false;
    Serial.println(String("Stepped to = ") + myStepper.currentPosition());
  }
  myStepper.run();
}


