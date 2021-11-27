import pandas as pd
import vaex


def sophos():
    #firewall
    print("START SOPHOS FIREWALL")
    df = vaex.from_json('sophosdeves.json', lines=True)
    df_pandas = df._source.to_pandas_series()
    dff1 = pd.json_normalize(df_pandas)

    df2 = dff1[['src_ip','dest_ip','src_country_code','dst_country_code','src_port','dest_port','bytes_out','bytes_in','user_name','action','device_name','protocol','log_component','log_type','device_vendor']]
    df2.to_csv (r'C:\Users\Emperor\Desktop\deveslog\v-insight-back\sophosdeves.csv')

    print("FINISH SOPHOS FIREWALL")
    return

def adaudit():
#adaudit
    print("START ADAUDIT")
    df = vaex.from_json('adauditplushdeves1h.json', lines=True)
    df_pandas = df._source.to_pandas_series()
    dff1 = pd.json_normalize(df_pandas)


    df2 = dff1[['src_ip', 'src_user', 'event_type_text', 'client_hostname', 'report_pro', 'ilogtype', 'event_number']]
    df2.to_csv (r'C:\Users\Emperor\Desktop\gitlab\v-insight-back\v-insight-back-sophos\adauditdeves.csv')

    print("FINISH ADAUDIT")
    return

def bluecoat():
    #bluecoat
    print("START BLUECOAT")
    df = vaex.from_json('bluecoatdeves1h.json', lines=True)
    df_pandas = df._source.to_pandas_series()
    dff1 = pd.json_normalize(df_pandas)

    df2 = dff1[['cs_host','src_ip','username','protocol','s_action','cs_categories','bytes_in','bytes_out']]
    df2.to_csv (r'C:\Users\Emperor\Desktop\gitlab\v-insight-back\v-insight-back-sophos\blucoatdeves.csv')

    print("FINISH BLUECOAT")
    return


def exchange():
    #exchange
    print("START EXCHANGE")
    df = vaex.from_json('exchange5days06.json', lines=True)
    df_pandas = df._source.to_pandas_series()
    dff1 = pd.json_normalize(df_pandas)

    df2 = dff1[['original_cli_ip','original_server_ip','server_hostname','ilogtype','message_subject','sender_address','recp_address','return_path','event_name','directionality','message_subject','recp_count']]
    df2.to_csv(r'C:\Users\Emperor\Desktop\gitlab\v-insight-back\v-insight-back-sophos\exchange.csv')

    print("FINISH EXCHANGE")




def adaudit1():
#adaudit
    print("START ADAUDIT")
    df = vaex.from_json('adauditplushdeves1h.json', lines=True)
    df_pandas = df._source.to_pandas_series()
    dff1 = pd.json_normalize(df_pandas)

    df2 = dff1[['src_ip','src_user','event_type_text','client_hostname','report_pro','ilogtype']]
    df2.to_csv (r'C:\Users\Emperor\Desktop\gitlab\v-insight-back\v-insight-back-sophos\adauditdeves1.csv')

    print("FINISH ADAUDIT")
    return



def fortigate():
    #firewall
    print("START FORTIGATE")
    df = vaex.from_json('ragnar-fortigate.json', lines=True)
    df_pandas = df._source.to_pandas_series()
    dff1 = pd.json_normalize(df_pandas)

    df2 = dff1[['src_ip','dest_ip','src_country','dest_country','src_port','dest_port','bytes_out','bytes_in','user','action','dev_name','protocol','logdesc']]
    df2.to_csv (r'C:\Users\Emperor\Desktop\gitlab\v-insight-back\v-insight-back-sophos\ragnar\fortigate-ragnar.csv')

    print("FINISH FORTIGATE")
    return

fortigate()

