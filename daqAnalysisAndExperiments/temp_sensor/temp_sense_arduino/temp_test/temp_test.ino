int ThermistorPin = 0, NUMSAMPLES = 5;
float V, T;
float b = 3477;
float T0 = 298.15;
int samples[5];

void setup() {
Serial.begin(9600);
}


void loop(void) {


V = analogRead(ThermistorPin);
T = 1 / ((1/b) * log(1/((float)V/1023) -1) + 1/T0) - 273.15 ;

 

delay(500);

  Serial.print("Temperature: "); 
  Serial.print(T);
  Serial.println(" C"); 
}

