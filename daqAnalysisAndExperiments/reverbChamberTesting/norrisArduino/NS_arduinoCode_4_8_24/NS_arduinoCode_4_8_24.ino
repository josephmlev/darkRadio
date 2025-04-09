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
bool start = false;
bool initialization = false;
bool relativeStepping = false;
bool zeroFlag = true;
bool stopFlag = true;
bool pwrCycleFlag = false; //when finding zero, commands a pc after stopping at true zero


void stopAndStabilize(){ // This stops motion of the stirrer, stabilizes the position with holding torque by switching the power on and off, and then sets position to zero
  //myStepper.stop();
  //myStepper.setSpeed(0); // if chop detected, stop motion
  //
  long stepsAfterChop = 480;
  long targetAfterChop = myStepper.currentPosition() + stepsAfterChop;

  myStepper.move(48);
  //

  //delay(1000);
  //digitalWrite(swPin, LOW);
  //delay(1000);
  //digitalWrite(swPin, HIGH);
  //delay(5000);
  //digitalWrite(swPin, LOW);
  //delay(1000);

  myStepper.setCurrentPosition(0); // sets the definite zero position
  timeElapsed = 0; // resetting time counter, just in case may be useful in future. timeElapsed starts counting at conclusion of startup
}


void moveToPosition(int pos){ 
  //desiredPosition = pos.toInt();
  Serial.println("Moving to position: " + String(pos));

  digitalWrite(swPin, HIGH);

  myStepper.moveTo(pos); // moveTo will move the stepper in ABSOLUTE coordinates to "pos" number. it goes from position 0 -> position "pos"
}


void setup() {
  Serial.begin(9600);
  myStepper.setMaxSpeed(400);    // must be equal to or greater than desired speed.
  myStepper.setAcceleration(40);
  //myStepper.setCurrentPosition(0);

  pinMode(swPin, OUTPUT);
  pinMode(encoderPin, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);


}

void loop() {

  if (Serial.available() > 0){ // this code runs if data is being sent via serial connection
    String command = Serial.readString();

    if (command.substring(0, 5) == "step "){ // command should be of form "step #"; causes stepper to step by "#" number of steps; "#" should not be entered in microsteps; stepper will NOT find zero first before moving
      stepNum = command.substring(5).toInt();
      stepNum = stepNum * microstepNum;
      start = false;
      //relativeStepping = true;
      digitalWrite(swPin, HIGH);
      delay(1000);
      myStepper.move(stepNum); //  will move the stepper in RELATIVE coordinates from (current position) -> (current position + stepNum)
      //timeElapsed = 0;
    }


    else if (command.substring(0, 9) == "position "){ // command should be of form "position #"; causes stepper to find zero, then move to "#" position; "#" should not be entered in microsteps
      stepNum = command.substring(9).toInt(); // logs step number for later
      stepNum = stepNum * microstepNum; // +3 accounts for the fact that there may be a consistent drift of 1 (or 3) microsteps when the stepper finds zero with chopper (current should be ~0.85A for non microstep holding power)
      start = true;
      digitalWrite(swPin, HIGH);
      timeElapsed = 0;
      initialization = true;
    }

    else if (command.substring(0, 4) == "zero"){ // makes stepper find zero using chopper (will not move to position after zeroing)
      stepNum = 0;
      start = true;
      digitalWrite(swPin, HIGH);
      delay(100);
      timeElapsed = 0;
      initialization = false;
    }

    else if(String(command) == "quit"){ // quit command disables power and closes serial connection
      digitalWrite(swPin, LOW);
      start = false;
      initialization = false;
      myStepper.moveTo(myStepper.currentPosition()); // come back to this, might be causing problems (pulses stop briefly then restart showing deceleration when quit ran)
      myStepper.stop();
      
      Serial.println("Sayonara!");
    }

    else if (command.substring(0, 5) == "power"){ // this is command to turn off relay (disables power to mode stirrer system), this will cause issues if ran when stepper motor is not in an integer position (in between positions)
      digitalWrite(swPin, LOW); // power can be restored by running any of the other commands (except quit) as they enable power by default.
      start = false;
      initialization = false;
      myStepper.stop();
      
      Serial.println("Power turning off...");
    }

    else if (command.substring(0, 2) == "pc"){ // This turns off and on the power to find the proper A+ zero
      delay(1000);
      digitalWrite(swPin, LOW);
      delay(1000);
      digitalWrite(swPin, HIGH);
      delay(1000);
      
      Serial.println("Power cycle complete. Power is off");
    }


    else if (command.substring(0, 6) == "getPos"){ // Query the position
      Serial.println(String("Current position = ") + myStepper.currentPosition());
    }

    else { // input not recognized
      Serial.println("Input not recognized, please try again, sir");
    }
  }
  encoderSignal = digitalRead(encoderPin);



  if(start == true){ // this code runs if the stepper is finding zero, it checks after every step taken to see if a chop is detected
    //myStepper.setSpeed(50); // move stepper at constant speed

    if(zeroFlag == true){
      Serial.println("here ");
      myStepper.setCurrentPosition(1); //prevents having to mod 
      myStepper.move(9999);
      zeroFlag = false;
    }
    
    if (encoderSignal == LOW and timeElapsed > (1 * 1000) and initialization == false){ // check if chop has been found within 5 seconds of startup, this code run if stepper is finding zero only
      chopPos = myStepper.currentPosition();
      myStepper.stop();
      //Serial.println(String("current Position ") + chopPos);
      start = false;
      stopFlag = true;
    }
    else if (encoderSignal == LOW and timeElapsed > (1 * 1000) and initialization == true){ // check if chop has been found within 5 seconds of startup, this code runs if stepper will move to position after finding zero
      stopAndStabilize();
      Serial.println("Zero position found. Motion stopped. Startup done. Now moving to desired position.");
      start = false;
      digitalWrite(swPin, HIGH);
      delay(1000);
      Serial.println("Now, moving to position: " + String(stepNum));
      myStepper.moveTo(stepNum);
      initialization = false;
    }
    else if(encoderSignal == HIGH and timeElapsed > (600000)){ // if 10 mins passes then there is a problem
      myStepper.stop();
      myStepper.setSpeed(0);
      start = false;
      digitalWrite(swPin, LOW);
      Serial.println("Stepper ran for 10 mins and did not detect chop. Check on system.");
    } 
    //myStepper.runSpeed(); // move at constant speed
    
  }
  if(start == false && stopFlag == true && !myStepper.isRunning()) { //it is stopped at chopPos past zero. Command move to true zero
    long stepsPastChop = myStepper.currentPosition() - chopPos;
    delay(1000);
    Serial.println("moving to true zero ");
    myStepper.move(4800 - stepsPastChop);
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
    Serial.println("Stopped at A+ zero");
    myStepper.setCurrentPosition(0);
    pwrCycleFlag = false;
  }


  myStepper.run();
  
}


