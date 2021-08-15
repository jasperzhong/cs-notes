LISTDIR = `ls -d */`

subdirs:
	for dir in $(LISTDIR); \
		do \
		$(MAKE) -C $$dir; \
		done

clean:
	for dir in $(LISTDIR); \
		do \
		$(MAKE) clean -C $$dir; \
		done
