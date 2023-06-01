using System;
using Python.Runtime;

class Apitest
{
    static void Main(string[] args)
    {   
        Runtime.PythonDLL = "/Users/coulson/opt/anaconda3/pkgs/python-3.9.13-hdfd78df_1/lib/libpython3.9.dylib";
        PythonEngine.Initialize();
        using (Py.GIL())
        {
            dynamic ssearchAPI = Py.Import("ssearch");
            ssearchAPI.init();  // Call init first
        }
    }
}
