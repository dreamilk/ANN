all:main.o sub.o
        gcc -o all main.o sub.o
main.o:main.c
        gcc -c main.c
sub.o:sub.c
        gcc -c sub.c
clean:
        rm main.o sub.o

