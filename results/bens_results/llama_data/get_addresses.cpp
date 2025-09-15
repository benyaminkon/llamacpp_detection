#include "pin.H"
#include <iostream>
#include <fstream>
#include <string>
#include <map>

using namespace std;

ofstream TraceFile;
ADDRINT libllama_base = 0;
bool libllama_found = false;
bool detokenize_active = false;
map<ADDRINT, string> detokenize_functions;

// Record memory reads ONLY during detokenize
VOID RecordMemRead(VOID *ip, VOID *addr, UINT32 size)
{
    if (!detokenize_active)
        return;

    ADDRINT read_addr = (ADDRINT)addr;
    TraceFile << "0x" << hex << read_addr << endl;
}

// Function entry - start tracing (no output to file)
VOID DetokenizeEntry(const string *func_name)
{
    detokenize_active = true;
}

// Function exit - stop tracing (no output to file)
VOID DetokenizeExit(const string *func_name)
{
    detokenize_active = false;
}

VOID Instruction(INS ins, VOID *v)
{
    if (!libllama_found)
        return;

    ADDRINT ins_addr = INS_Address(ins);
    ADDRINT offset = ins_addr - libllama_base;

    // Check for function entry
    if (detokenize_functions.count(offset) > 0)
    {
        string func_name = detokenize_functions[offset];
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)DetokenizeEntry,
                       IARG_PTR, new string(func_name), IARG_END);
    }

    // Instrument memory reads
    if (INS_IsMemoryRead(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
            IARG_INST_PTR,
            IARG_MEMORYREAD_EA,
            IARG_MEMORYREAD_SIZE,
            IARG_END);
    }

    // Detect function exits
    if (INS_IsRet(ins) && ins_addr >= libllama_base && ins_addr < libllama_base + 0x200000)
    {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)DetokenizeExit,
                       IARG_PTR, new string("detokenize_function"), IARG_END);
    }
}

VOID ImageLoad(IMG img, VOID *v)
{
    string img_name = IMG_Name(img);

    if (img_name.find("libllama.so") != string::npos)
    {
        libllama_base = IMG_LowAddress(img);
        libllama_found = true;

        // Target only llama_token_to_piece
        detokenize_functions[0x16a4e0] = "llama_token_to_piece";
    }
}

VOID Fini(INT32 code, VOID *v)
{
    TraceFile.close();
}

INT32 Usage()
{
    cerr << "Clean address tracer" << endl;
    return -1;
}

int main(int argc, char *argv[])
{
    if (PIN_Init(argc, argv))
        return Usage();

    TraceFile.open("detokenize_trace.out");

    IMG_AddInstrumentFunction(ImageLoad, 0);
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    PIN_StartProgram();
    return 0;
}