
typedef enum State{SEND_HANDSHAKE, RECEIVE_HANDSHAKE, SEND_DATA, CONFIRM_DATA};

State currentState;
char startChar = '<';
char endChar = '>';
char abortChar = '^';
unsigned long currentTime;
const unsigned long MAX_DELTA = 1000;

int ThermistorPin = 0;
float V, T;
float b = 3477;
float T0 = 298.15;
float samples[5];

int totalSent = 0;
int maxTries = 3;

void serialFlush()
{
  while(Serial.available() > 0) 
  {
    char t = Serial.read();
  }
  
}
String getData()
{
  int i;
  for (i=0; i< 5; i++) {
    digitalWrite(13, HIGH);
    V = analogRead(ThermistorPin);
    digitalWrite(13, LOW);
    samples[i] = 1 / ((1/b) * log(1/((float)V/1023) -1) + 1/T0) - 273.15 ;
    delay(20);
  }
 
  // average all the samples out
  T = 0;
  for (i=0; i< 5; i++) {
     T += samples[i];
  }
  T /= 5;
  return String(T);
}

void setup() 
{
    Serial.begin(9600);
    currentState = SEND_HANDSHAKE;
    currentTime = millis();
}
void loop()
{
  switch(currentState)
  {
     case SEND_HANDSHAKE:
      Serial.print(startChar);
      Serial.print('\n');
      currentState = RECEIVE_HANDSHAKE;
      currentTime = millis();
     case RECEIVE_HANDSHAKE:
      char rc;
      if(millis() - currentTime > MAX_DELTA)
      {
        currentState = SEND_HANDSHAKE;
        currentTime = millis();
      }
      else
      {
       while (Serial.available() && (millis() - currentTime) < MAX_DELTA)
       {
        rc = Serial.read();
        if(rc == endChar)
        {
          currentState = SEND_DATA;
          currentTime = millis();
          break;
        }
       }
      } 
     case SEND_DATA:
      Serial.print(getData());
      Serial.print('\n');
      currentTime = millis();
      currentState = CONFIRM_DATA;
     case CONFIRM_DATA:
      char rc2;
      if(millis() - currentTime > MAX_DELTA)
      {          
        currentState = SEND_DATA;
        currentTime = millis();
        totalSent = totalSent + 1;
        if(totalSent > maxTries)
        {
          totalSent = 0;
          currentState = SEND_HANDSHAKE;
        }
      }
      else
      {
       while (Serial.available() && (millis() - currentTime) < MAX_DELTA)
       {
        rc2 = Serial.read();
        if(rc2 == endChar)
        {
          currentState = SEND_HANDSHAKE;
          currentTime = millis();
          break;
        }
       }
       currentState = SEND_HANDSHAKE;
      }
      delay(20);
  }
}
