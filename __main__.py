
import os, sys

_ipath, _iname = os.path.split(os.path.dirname(__file__))
if _ipath not in sys.path:
	sys.path.insert(0, _ipath)
pkg = __import__(_iname)
del _ipath, _iname

def main():
	pkg.main()

if __name__ == '__main__':
	main()
