all: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX
all: LIBS += $(shell pkg-config --libs libadq)
all: adqapiexample
debug: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX
debug: LIBS += $(shell pkg-config --libs libadq)

local: CFLAGS += -DLINUX -Wl,-rpath=../../source/.libs -g
local: LIBS += -ladq -ludev -lpthread -lstdc++
local:example*.c
	$(CC) $(CFLAGS) -I../../Release -o adqapiexample example*.c -L../../source/.libs $(LIBS)

debug:example*.c
	$(CC) $(CFLAGS) -g -o adqapiexample example*.c $(LIBS)

adqapiexample:example*.c
	$(CC) $(CFLAGS) -o adqapiexample example*.c $(LIBS)

clean:
	rm -f adqapiexample
