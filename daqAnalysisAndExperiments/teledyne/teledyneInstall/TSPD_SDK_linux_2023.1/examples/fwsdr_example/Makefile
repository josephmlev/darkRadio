all: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra -Wpedantic
all: LIBS += $(shell pkg-config --libs libadq)
all: adqapifwsdrexample
debug: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra -Wpedantic
debug: LIBS += $(shell pkg-config --libs libadq)

local: CFLAGS += -DLINUX -Wall -Wextra -Wpedantic
local: LIBS += -ladq -ludev -lpthread -lstdc++
local:ADQAPI_FWSDR_example.c
	$(CC) $(CFLAGS) -I../../Release -o adqapifwsdrexample ADQAPI_FWSDR_example.c -L../../source/.libs $(LIBS)

debug:ADQAPI_FWSDR_example.c
	$(CC) $(CFLAGS) -g -o adqapifwsdrexample ADQAPI_FWSDR_example.c $(LIBS)

adqapifwsdrexample:ADQAPI_FWSDR_example.c
	$(CC) $(CFLAGS) -o adqapifwsdrexample ADQAPI_FWSDR_example.c $(LIBS)

clean:
	rm -f adqapifwsdrexample
