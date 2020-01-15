#pragma once
#ifdef DEBUG
#define IVE_DEBUG(fmt, args...) \
  fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
#define IVE_DEBUG(fmt, args...)
#endif