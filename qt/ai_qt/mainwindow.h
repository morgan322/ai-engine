#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QComboBox>
#include <QPlainTextEdit>
#include <QVBoxLayout>
#include <QJsonObject>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

private:
    QComboBox* m_timeZoneCombo;
    QPlainTextEdit* m_logEdit;
    QComboBox* m_dataSourceCombo;
    QComboBox* m_detectionBoxCombo;
    QComboBox* m_logLevelCombo;
    QPlainTextEdit* m_resultEdit;

    QVBoxLayout* m_systemLayout;
    QVBoxLayout* m_algorithmLayout;
    QVBoxLayout* m_logLayout;
    QVBoxLayout* m_mainLayout;

    QJsonObject m_configData;

    void saveConfigToJson();

private slots:
    void saveParameter();
    void printLogAndResult();
};

#endif // MAINWINDOW_H
