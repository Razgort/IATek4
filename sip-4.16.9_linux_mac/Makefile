
all:
	@(cd sipgen; $(MAKE))
	@(cd siplib; $(MAKE))

install:
	@(cd sipgen; $(MAKE) install)
	@(cd siplib; $(MAKE) install)
	@test -d $(DESTDIR)/Library/Python/2.7/site-packages || mkdir -p $(DESTDIR)/Library/Python/2.7/site-packages
	cp -f sipconfig.py $(DESTDIR)/Library/Python/2.7/site-packages/sipconfig.py
	cp -f /Users/Orel/OCR/ocr/sip-4.16.9_linux_mac/sipdistutils.py $(DESTDIR)/Library/Python/2.7/site-packages/sipdistutils.py

clean:
	@(cd sipgen; $(MAKE) clean)
	@(cd siplib; $(MAKE) clean)
