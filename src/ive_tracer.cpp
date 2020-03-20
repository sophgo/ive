#include "ive_tracer.h"

#include "tracer/tracer.h"

void CVI_TraceBegin(const char *name) { Tracer::TraceBegin(name); }

void CVI_TraceCounter(const char *name, signed int value) { Tracer::TraceCounter(name, value); }

void CVI_TraceEnd() { Tracer::TraceEnd(); }