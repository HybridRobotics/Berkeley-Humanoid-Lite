// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once

#define SET_BITS(REG, BIT)                    ((REG) |= (BIT))
#define CLEAR_BITS(REG, BIT)                  ((REG) &= ~(BIT))
#define READ_BITS(REG, BIT)                   ((REG) & (BIT))
#define WRITE_BITS(REG, CLEARMASK, SETMASK)   ((REG) = ((REG) & ~(CLEARMASK)) | (SETMASK))
