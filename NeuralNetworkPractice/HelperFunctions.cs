using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPractice
{


    class HelperFunctions
    {
        public static List<T> ListColumn<T>(List<List<T>> list, int column)
        {
            return list.ConvertAll((x => x[column]));
        }

        public static string ListToString<T>(List<T> list)
        {
            return string.Join(" ", list.ConvertAll(x => x.ToString()));
        }

        public static string ListToString<T>(List<List<T>> list)
        {
            return string.Join("\n", list.ConvertAll(x => ListToString(x)));
        }
    }
}
