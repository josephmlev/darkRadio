CFLAGS += $(shell pkg-config --cflags libadq) -DLINUX
LIBS += $(shell pkg-config --libs libadq)
SOURCES = formatter.c \
    ADQAPI_FWPD_example.c \
    example_adq14.c \
    example_adq7.c

OBJECTS = $(SOURCES:.c=.o)

local: CFLAGS += -I../../Release
local: LDFLAGS += -L../../source/.libs
debug: CFLAGS += -g

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

adqapifwpdexample: $(OBJECTS)
	$(CC) $(CFLAGS) -o adqapifwpdexample $(OBJECTS) $(LDFLAGS) $(LIBS)

all local debug: adqapifwpdexample

clean:
	rm -f adqapifwpdexample $(OBJECTS)
