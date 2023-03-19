all: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra -Wpedantic
all: LIBS += $(shell pkg-config --libs libadq)
all: adqapitransfertest
debug: CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX -Wall -Wextra -Wpedantic
debug: LIBS += $(shell pkg-config --libs libadq)

local: CFLAGS += -DLINUX -Wall -Wextra -Wpedantic
local: LIBS += -ladq -ludev -lpthread -lstdc++
local:example*.c
	$(CC) $(CFLAGS) -I../../Release -o adqapitransfertest example*.c -L../../source/.libs $(LIBS)

debug:example*.c
	$(CC) $(CFLAGS) -g -o adqapitransfertest example*.c $(LIBS)

adqapitransfertest:example*.c
	$(CC) $(CFLAGS) -o adqapitransfertest example*.c $(LIBS)

clean:
	rm -f adqapitransfertest
