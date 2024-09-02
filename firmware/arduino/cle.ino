int led = 13;
int val = -1;
int count = 0;
int flag_firsttime = 1;

void setup() {
  // put your setup code here, to run once:
  pinMode(led, OUTPUT);
  digitalWrite(led, LOW);
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);  
  digitalWrite(7, LOW);
  digitalWrite(8, LOW);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(led, HIGH);

  if(flag_firsttime == 1){
    flag_firsttime = 0;
    delay(10000);
    digitalWrite(7, HIGH);
    delay(1000);
    digitalWrite(8, HIGH);
    delay(1000);
    digitalWrite(8, LOW);
  }


// Delay/wait for a second (1000 milliseconds = 1 second)
  delay(1000);

// Turn the LED off (LOW is “off” in the Arduino language)
  digitalWrite(led, LOW);
// Delay/wait for a second
  delay(1000);
  count = count + 1;
}
