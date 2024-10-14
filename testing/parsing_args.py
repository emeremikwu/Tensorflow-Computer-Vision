import argparse
import sys


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('infile', nargs='?')
  parser.add_argument('outfile', nargs='?')
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('--label', nargs='2')
  parser.parse_args(['input.txt', 'output.txt'])

  args = parser.parse_args(["--label" "hello"])

  print(args)
