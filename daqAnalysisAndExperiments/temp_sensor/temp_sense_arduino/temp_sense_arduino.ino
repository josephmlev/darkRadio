int ThermistorPin = 0;
float V, T;
float b = 3477;
float T0 = 298.15;
float samples[5];

void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT); 
  analogReference(EXTERNAL);
}

// Take 5 samples in a row and average them. Toggle power
// on only when taking a sample as to not warm thermistor
void loop(void) {
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
  
  delay(1000);
  Serial.println(T);
  
}

