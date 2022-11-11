import paraview.simple as ps
import paraview.servermanager as psm



def main():

    psm.Disconnect()
    connection = psm.ActiveConnection
    psm.Connect()
    connection = psm.ActiveConnection
    print(connection.GetURI())



if __name__ == "__main__":

    main()
