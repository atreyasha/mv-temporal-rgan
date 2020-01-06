#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

class arg_metav_formatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    """
    Class to combine argument parsers in order to display meta-variables
    and defaults for arguments
    """
    pass
